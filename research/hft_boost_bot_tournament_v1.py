from __future__ import annotations
import argparse, json, math
from collections import deque
from decimal import Decimal
from pathlib import Path
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.hft_boost_raw_xau_bt import HFTBaseConfig, metrics
from research.hft_boost_raw_xau_bt_event import HFTBaseEventStrategy, _price_float, _money_float

SIM=Venue('SIM')

class RawBarBuilder:
    def __init__(self, seconds:int, maxlen=500): self.seconds=seconds; self.bars=deque(maxlen=maxlen); self.cur=None
    def update(self, ts_ms, bid, ask):
        px=(bid+ask)/2.0; bucket=(ts_ms//(self.seconds*1000))*(self.seconds*1000)
        if self.cur is None:
            self.cur={'ts':bucket,'open':px,'high':px,'low':px,'close':px,'ticks':1}; return False
        if bucket!=self.cur['ts']:
            self.bars.append(self.cur); self.cur={'ts':bucket,'open':px,'high':px,'low':px,'close':px,'ticks':1}; return True
        self.cur['high']=max(self.cur['high'],px); self.cur['low']=min(self.cur['low'],px); self.cur['close']=px; self.cur['ticks']+=1; return False
    def data(self): return list(self.bars)

def ema(vals, n):
    if len(vals)<n: return None
    a=2/(n+1); x=vals[0]
    for v in vals[1:]: x=a*v+(1-a)*x
    return x

def rsi(vals,n=14):
    if len(vals)<n+1:return None
    d=[vals[i]-vals[i-1] for i in range(1,len(vals))][-n:]; g=sum(max(x,0) for x in d)/n; l=sum(max(-x,0) for x in d)/n
    return 100.0 if l==0 else 100-100/(1+g/l)

def atr(bars,n=14):
    if len(bars)<n+1:return None
    tr=[]
    for i in range(1,len(bars)):
        b=bars[i]; pc=bars[i-1]['close']; tr.append(max(b['high']-b['low'],abs(b['high']-pc),abs(b['low']-pc)))
    return sum(tr[-n:])/n

def stoch_k(bars,n=8):
    if len(bars)<n:return None
    w=bars[-n:]; lo=min(x['low'] for x in w); hi=max(x['high'] for x in w); c=w[-1]['close']; return 50.0 if hi==lo else (c-lo)/(hi-lo)*100

def macd_hist(vals):
    if len(vals)<35:return None
    def series(n):
        a=2/(n+1); out=[]; x=vals[0]
        for v in vals: x=a*v+(1-a)*x; out.append(x)
        return out
    e12,e26=series(12),series(26); m=[a-b for a,b in zip(e12,e26)]; a=2/10; sig=[]; x=m[0]
    for v in m: x=a*v+(1-a)*x; sig.append(x)
    return m[-1]-sig[-1], m[-2]-sig[-2]

def swing_structure(bars, strength=2, lookback=40):
    x=bars[-lookback:]; hs=[]; ls=[]
    for i in range(strength,len(x)-strength):
        if x[i]['high']>max(b['high'] for b in x[i-strength:i]) and x[i]['high']>max(b['high'] for b in x[i+1:i+1+strength]): hs.append(x[i]['high'])
        if x[i]['low']<min(b['low'] for b in x[i-strength:i]) and x[i]['low']<min(b['low'] for b in x[i+1:i+1+strength]): ls.append(x[i]['low'])
    trend='NEUTRAL'
    if len(hs)>=2 and len(ls)>=2:
        if hs[-1]>hs[-2] and ls[-1]>ls[-2]: trend='UP'
        elif hs[-1]<hs[-2] and ls[-1]<ls[-2]: trend='DOWN'
    return trend,(hs[-1] if hs else None),(ls[-1] if ls else None)

class BotTournament(HFTBaseEventStrategy):
    def __init__(self, config, bot):
        super().__init__(config); self.bot=bot; self.m1=RawBarBuilder(60); self.m5=RawBarBuilder(300); self.m15=RawBarBuilder(900)
        self.entry_spread=0.0; self.tp=0.0; self.sl=0.0; self.maxp=0.0; self.last_signal_bar=-1
        self.exit_counts={'tp':0,'sl':0,'time':0,'peak':0}; self.profile_rejects=0
    def _update_bars(self,ts,bid,ask):
        a=self.m1.update(ts,bid,ask); self.m5.update(ts,bid,ask); self.m15.update(ts,bid,ask); return a
    def _gold(self,m):
        ts,bid,ask,vel,imb,spread,exh=m; b1=self.m1.data(); b5=self.m5.data(); b15=self.m15.data()
        if len(b1)<30:return None
        a=atr(b1,14); score=0; side=None
        last=b1[-1]; body=abs(last['close']-last['open']); sign=1 if last['close']>last['open'] else -1
        if a and body>=1.8*a and abs(imb)>=35 and abs(vel)>=8 and ((vel>0)==(sign>0)):
            side='buy' if sign>0 else 'sell'; score=70+min(15,(body/a-1.8)*8)
        if side is None and len(b5)>=25 and len(b15)>=25:
            c5=[b['close'] for b in b5]; c15=[b['close'] for b in b15]; e9=ema(c5,9); e20=ema(c15,20)
            if e9 and e20:
                tr_up=c15[-1]>e20; tr_dn=c15[-1]<e20
                if abs(c5[-1]-e9)<=max(0.025,a*0.25 if a else 0.025): side='buy' if tr_up else ('sell' if tr_dn else None); score=62
        if side is None:return None
        if spread>830:return None
        score += 5 if spread<=630 else 0
        score += min(12,abs(imb)*0.2) + min(8,abs(vel)*0.25)
        return (side,score) if score>=60 else None
    def _roy(self,m):
        if len(self.m1.data())<60 or len(self.m15.data())<60:return None
        b1,b15=self.m1.data(),self.m15.data(); c1=[b['close'] for b in b1]; c15=[b['close'] for b in b15]
        e20,e50,e20h,e50h=ema(c1,20),ema(c1,50),ema(c15,20),ema(c15,50); rr=rsi(c1,14); k=stoch_k(b1,8)
        prevk=stoch_k(b1[:-1],8) if len(b1)>9 else None; spread=m[5]
        if None in (e20,e50,e20h,e50h,rr,k,prevk) or spread>760:return None
        vol=b1[-1]['ticks']
        if vol<50:return None
        buy=e20h>e50h and e20>e50 and rr>50 and k>prevk and k<80
        sell=e20h<e50h and e20<e50 and rr<50 and k<prevk and k>20
        if not(buy or sell):return None
        return ('buy' if buy else 'sell',75)
    def _testbot(self,m):
        b=self.m1.data()
        if len(b)<60:return None
        c=[x['close'] for x in b]; ef=ema(c,9); es=ema(c,21); a=atr(b,14)
        if None in (ef,es,a):return None
        prevc=[x['close'] for x in b[:-1]]; pef=ema(prevc,9); pes=ema(prevc,21)
        if pef<=pes and ef>es:return ('buy',72)
        if pef>=pes and ef<es:return ('sell',72)
        mom=c[-1]/c[-4]-1 if len(c)>=4 else 0
        if ef>es and mom>0.00015:return ('buy',68)
        if ef<es and mom<-0.00015:return ('sell',68)
        if ef>es and b[-1]['low']<=pef:return ('buy',64)
        if ef<es and b[-1]['high']>=pef:return ('sell',64)
        return None
    def _midas(self,m):
        b=self.m5.data(); h=self.m15.data()
        if len(b)<45 or len(h)<30:return None
        c=[x['close'] for x in b]; hist=macd_hist(c); a=atr(b,14); trend,sh,sl=swing_structure(b); htf,_sh,_sl=swing_structure(h)
        if hist is None or a is None:return None
        now=b[-1]; buy=[]; sell=[]
        if hist[1]<0<hist[0]:buy.append('macd')
        elif hist[1]>0>hist[0]:sell.append('macd')
        if hist[0]>hist[1]>0:buy.append('mom')
        elif hist[0]<hist[1]<0:sell.append('mom')
        if trend=='UP':buy.append('structure')
        elif trend=='DOWN':sell.append('structure')
        if sh and now['high']>sh and now['close']<sh and now['close']<now['open']:sell.append('liq')
        if sl and now['low']<sl and now['close']>sl and now['close']>now['open']:buy.append('liq')
        if htf=='UP':buy.append('htf')
        elif htf=='DOWN':sell.append('htf')
        if len(buy)>=3 and len(buy)>len(sell) and htf!='DOWN':return ('buy',60+5*len(buy))
        if len(sell)>=3 and len(sell)>len(buy) and htf!='UP':return ('sell',60+5*len(sell))
        return None
    def _profile_signal(self,m):
        return {'gold':self._gold,'roy':self._roy,'testbot':self._testbot,'midas':self._midas}[self.bot](m)
    def on_quote_tick(self,tick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); ts=int(tick.ts_event//1_000_000); newm1=self._update_bars(ts,bid,ask)
        m=self._micro(tick)
        if m is None:return
        if self.entry_price is not None and not self.exit_pending:
            signed=1 if self.entry_side=='buy' else -1; mark=bid if self.entry_side=='buy' else ask; d=signed*(mark-self.entry_price)/self.config.point; held=ts-self.entry_ts_ms; self.maxp=max(self.maxp,d)
            reason=None
            if d>=self.tp: reason='tp'
            elif d<=-self.sl: reason='sl'
            elif self.maxp>=self.entry_spread*1.0 and d<=self.maxp-self.entry_spread*0.75: reason='peak'
            elif held>=60000: reason='time'
            if reason:
                self.exit_counts[reason]+=1; self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
        if self.entry_price is not None or self.entry_pending or self.exit_pending:return
        if ts-self.last_exit_ts_ms<self.config.cooldown_ms:return
        if self.bot!='gold' and not newm1:return
        s=self._profile_signal(m)
        if s is None:return
        side,score=s; spread=m[5]
        a=atr(self.m1.data(),14) or (spread*self.config.point)
        atr_pts=max(a/self.config.point, spread)
        if self.bot=='roy': self.sl=max(spread*1.5,atr_pts*1.5); self.tp=max(spread*2.5,self.sl*2.5)
        elif self.bot=='testbot': self.sl=max(spread*1.5,atr_pts*1.2); self.tp=max(spread*2.0,atr_pts*1.8)
        elif self.bot=='midas': self.sl=max(spread*1.5,atr_pts*1.0); self.tp=max(spread*2.0,self.sl*2.0)
        else: self.sl=max(spread*1.5,atr_pts*1.2); self.tp=max(spread*2.0,atr_pts*1.5)
        inst=self.cache.instrument(self.config.instrument_id); order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side=='buy' else OrderSide.SELL,quantity=inst.make_qty(self.config.trade_size))
        self.entry_spread=spread; self.maxp=0; self.pending_side=side; self.entry_pending=True; self.entries+=1; self.signal_count+=1; self.score_sum+=score; self.submit_order(order)
    def on_position_closed(self,event):
        pnl=_money_float(getattr(event,'realized_pnl',None)); ts=getattr(event,'ts_closed',None) or getattr(event,'ts_event',0); self.closed_trades.append({'pnl':pnl,'ts_closed':int(ts or 0)})
        self._clear_entry_pending(); self.last_exit_ts_ms=int(getattr(event,'ts_event',0)//1_000_000); self.entry_price=None; self.entry_side=None; self.entry_ts_ms=None; self.exit_pending=False

def query_ticks(catalog, instrument_id):
    try:return catalog.query(QuoteTick,identifiers=[instrument_id])
    except TypeError:return catalog.query(QuoteTick,instrument_ids=[instrument_id])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--bot',choices=['gold','roy','testbot','midas'],required=True); ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--initial',type=float,default=1000); ap.add_argument('--trade-size',default='1'); args=ap.parse_args()
    if not args.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); days=int(manifest['days']); cat=ParquetDataCatalog(str(cp)); inst={x.id.symbol.value.replace('/',''):x for x in cat.instruments()}.get('XAUUSD'); ticks=query_ticks(cat,inst.id.value)
    point=float(inst.price_increment.as_double()) if hasattr(inst.price_increment,'as_double') else float(inst.price_increment)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True))); eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(args.initial,USD)],default_leverage=Decimal('2000')); eng.add_instrument(inst); eng.add_data(ticks)
    strat=BotTournament(HFTBaseConfig(instrument_id=inst.id,trade_size=Decimal(args.trade_size),point=point,min_score=0,tp_points=8,sl_points=10,cooldown_ms=250),args.bot); eng.add_strategy(strat); eng.run(); trades=strat.closed_trades; k=metrics(trades,args.initial,days)
    out=Path('results/ae-bt')/args.experiment_id; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(trades).to_csv(out/'trades.csv',index=False)
    summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','edge':f'BOT_TOURNAMENT_{args.bot}_v1','bot_profile':args.bot,'port_policy':'source-faithful adapter; Raw QuoteTick aggregation to in-memory bars; execution stays native Bid/Ask QuoteTick','ohlc_resample_used':False,'raw_ticks':len(ticks),'point':point,'signals':strat.signal_count,'entries_submitted':strat.entries,'order_fills':strat.order_fills,'order_rejects':strat.order_rejects,'closed_positions':len(trades),'exit_counts':strat.exit_counts,'kpi':k,'limitations':['This is a source-faithful Nautilus port, not byte-identical MT5 execution.','No explicit commission/slippage yet; native Bid/Ask spread is present.']}; txt=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True); (out/'summary.json').write_text(txt); print(txt); eng.dispose()
if __name__=='__main__':main()
