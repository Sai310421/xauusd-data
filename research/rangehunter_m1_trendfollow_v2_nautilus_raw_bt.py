from __future__ import annotations
import argparse,json
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
SIM=Venue('SIM')
P=dict(ema=21,atr=14,adx=14,max_adx=32.0,bb=20,bb_dev=2.0,max_bb_width_atr=4.5,
       pullback_atr=0.28,min_body_atr=0.05,sl_atr=0.70,tp_r=1.40,be_r=0.45,trail_start_r=0.75,
       trail_atr=0.45,risk_pct=0.50,max_lot=2.0,max_spread=1.20,start_hour=6,end_hour=23,
       friday_stop_hour=20,max_hold_sec=180,cooldown_sec=30)
class Config(StrategyConfig,frozen=True):
    instrument_id:InstrumentId
    bar_type:BarType
class Strat(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg); self.b=deque(maxlen=180); self.armed=None; self.entry=None; self.stop_ref=None
        self.risk=None; self.tp=None; self.side=None; self.entry_ts=None; self.exit_pending=False; self.baskets=0
        self.setup_count=0; self.direction_count=0; self.pullback_count=0; self.cool_until=0
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def arr(self,k): return np.array([x[k] for x in self.b],float)
    def ema(self,x,n):
        if len(x)<n:return None
        a=2/(n+1); v=float(np.mean(x[:n]))
        for z in x[n:]:v=a*float(z)+(1-a)*v
        return v
    def atr(self,n=14):
        if len(self.b)<n+1:return None
        x=list(self.b); tr=[max(x[i]['h']-x[i]['l'],abs(x[i]['h']-x[i-1]['c']),abs(x[i]['l']-x[i-1]['c'])) for i in range(1,len(x))]
        a=sum(tr[:n])/n
        for z in tr[n:]:a=(a*(n-1)+z)/n
        return float(a)
    def adx(self,n=14):
        x=list(self.b)
        if len(x)<2*n+2:return None
        tr=[];pd=[];md=[]
        for i in range(1,len(x)):
            up=x[i]['h']-x[i-1]['h'];dn=x[i-1]['l']-x[i]['l'];pd.append(up if up>dn and up>0 else 0);md.append(dn if dn>up and dn>0 else 0)
            tr.append(max(x[i]['h']-x[i]['l'],abs(x[i]['h']-x[i-1]['c']),abs(x[i]['l']-x[i-1]['c'])))
        at=sum(tr[:n]);ps=sum(pd[:n]);ms=sum(md[:n]);dx=[]
        for i in range(n,len(tr)):
            if i>n:at=at-at/n+tr[i];ps=ps-ps/n+pd[i];ms=ms-ms/n+md[i]
            p=100*ps/at if at else 0;m=100*ms/at if at else 0;dx.append(100*abs(p-m)/(p+m) if p+m else 0)
        if len(dx)<n:return None
        a=sum(dx[:n])/n
        for z in dx[n:]:a=(a*(n-1)+z)/n
        return float(a)
    def snap(self):
        if len(self.b)<50:return None
        c=self.arr('c'); atr=self.atr(); adx=self.adx(); ema=self.ema(c,P['ema'])
        if atr is None or adx is None or ema is None or atr<=0:return None
        w=c[-P['bb']:];mid=float(np.mean(w));sd=float(np.std(w));width=2*P['bb_dev']*sd
        prev=self.ema(c[:-1],P['ema'])
        return dict(atr=atr,adx=adx,ema=ema,ema_prev=prev,width=width,mid=mid)
    def on_bar(self,bar:Bar):
        x={'o':self.f(bar.open),'h':self.f(bar.high),'l':self.f(bar.low),'c':self.f(bar.close),'ts':int(bar.ts_event)};self.b.append(x)
        if self.entry is not None:return
        s=self.snap()
        if s is None:return
        self.setup_count+=1
        if s['adx']>P['max_adx'] or s['width']>P['max_bb_width_atr']*s['atr']:return
        body=abs(x['c']-x['o'])
        buy_dir=x['c']>s['ema'] and s['ema']>=s['ema_prev'];sell_dir=x['c']<s['ema'] and s['ema']<=s['ema_prev']
        if buy_dir or sell_dir:self.direction_count+=1
        buy_pull=x['l']<=s['ema']+P['pullback_atr']*s['atr'] and x['c']>s['ema']
        sell_pull=x['h']>=s['ema']-P['pullback_atr']*s['atr'] and x['c']<s['ema']
        if (buy_dir and buy_pull) or (sell_dir and sell_pull):self.pullback_count+=1
        buy=buy_dir and buy_pull and x['c']>x['o'] and body>=P['min_body_atr']*s['atr']
        sell=sell_dir and sell_pull and x['c']<x['o'] and body>=P['min_body_atr']*s['atr']
        if buy:self.armed=('BUY',s)
        elif sell:self.armed=('SELL',s)
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);dt=pd.Timestamp(ts,unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed and self.entry is None and flat and ts>=self.cool_until:
            if ask-bid>P['max_spread'] or not(P['start_hour']<=dt.hour<P['end_hour']) or (dt.weekday()==4 and dt.hour>=P['friday_stop_hour']):return
            side,s=self.armed;px=ask if side=='BUY' else bid;r=P['sl_atr']*s['atr'];money=1000*P['risk_pct']/100;total=min(P['max_lot']*100,money/r if r>0 else 0)
            if total<=0:return
            instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if side=='BUY' else OrderSide.SELL
            for frac in (0.50,0.30,0.20):
                q=instr.make_qty(Decimal(str(total*frac)));self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=q))
            self.entry=px;self.risk=r;self.stop_ref=px-r if side=='BUY' else px+r;self.tp=px+P['tp_r']*r if side=='BUY' else px-P['tp_r']*r
            self.side=side;self.entry_ts=ts;self.exit_pending=False;self.baskets+=1;self.armed=None;return
        if self.entry is None or self.exit_pending:return
        px=bid if self.side=='BUY' else ask;profit=px-self.entry if self.side=='BUY' else self.entry-px;R=profit/self.risk if self.risk else 0;s=self.snap();atr=s['atr'] if s else self.risk/P['sl_atr']
        if R>=P['be_r']:
            be=self.entry+(0.02 if self.side=='BUY' else -0.02);self.stop_ref=max(self.stop_ref,be) if self.side=='BUY' else min(self.stop_ref,be)
        if R>=P['trail_start_r']:
            ns=px-P['trail_atr']*atr if self.side=='BUY' else px+P['trail_atr']*atr;self.stop_ref=max(self.stop_ref,ns) if self.side=='BUY' else min(self.stop_ref,ns)
        hit=(px<=self.stop_ref or px>=self.tp) if self.side=='BUY' else (px>=self.stop_ref or px<=self.tp)
        timeout=(ts-self.entry_ts)>=P['max_hold_sec']*1_000_000_000
        if hit or timeout:self.close_all_positions(self.config.instrument_id);self.exit_pending=True
    def on_position_closed(self,e):
        self.cool_until=int(e.ts_event)+P['cooldown_sec']*1_000_000_000;self.entry=self.stop_ref=self.risk=self.tp=self.side=self.entry_ts=None;self.exit_pending=False
    def on_stop(self):self.close_all_positions(self.config.instrument_id)
def money(v):
    try:return float(str(v).replace(',','').split()[0])
    except:return 0.0
def extract(r):
    if r is None or r.empty:return []
    pc=next((c for c in r.columns if 'pnl' in str(c).lower()),None);tc=next((c for c in r.columns if 'closed' in str(c).lower()),None);out=[]
    for i,row in r.iterrows():
        try:ts=int(pd.Timestamp(row[tc]).value) if tc else i
        except:ts=i
        out.append({'symbol':'XAUUSD','tf':'M1','pnl':money(row[pc]) if pc else 0.0,'ts_closed':ts})
    return out
def metrics(t,initial=1000.0,days=30):
    a=np.array([x['pnl'] for x in t],float)
    if not len(a):return dict(N=0,WR_pct=0,PF=0,NetProfit=0,MaxDD_pct=0,RF=None,Monthly21_pct=0,Daily_pct=0,MaxConsecutiveLosses=0)
    w=a[a>0].sum();l=abs(a[a<0].sum());eq=peak=initial;mdd=0;run=mx=0
    for z in a:
        eq+=z;peak=max(peak,eq);mdd=max(mdd,peak-eq);run=run+1 if z<0 else 0;mx=max(mx,run)
    net=float(a.sum());dd=100*mdd/peak if peak else 0;m21=((max(eq,1e-9)/initial)**(21/days)-1)*100;daily=((1+m21/100)**(1/21)-1)*100
    return dict(N=len(a),WR_pct=float((a>0).mean()*100),PF=float(w/l) if l else None,NetProfit=net,MaxDD_pct=dd,RF=net/mdd if mdd else None,Monthly21_pct=m21,Daily_pct=daily,MaxConsecutiveLosses=mx)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run();tr=extract(eng.trader.generate_positions_report());met=metrics(tr,days=int(man['days']))
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);summary=dict(verification_level='NAUTILUS_BT_RAW_BIDASK',engine='NautilusTrader BacktestEngine',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),strategy='RangeHunter_M1_TrendFollow_v2',data_kind='RAW_BIDASK QuoteTick',ohlc_resample_used=False,signal_bars='Nautilus INTERNAL 1-MINUTE BID bars',execution='3 market entries 50/30/20 on raw QuoteTicks; observed spread native',period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),raw_tick_count=len(ticks),diagnostics=dict(bar_setups=st.setup_count,direction_pass=st.direction_count,pullback_pass=st.pullback_count,baskets_submitted=st.baskets),params=P,metrics=met,limitations=['UTC session mapping','0.5% initial-equity risk model; max 2 lots=200 oz','No explicit commission or probabilistic slippage beyond native Bid/Ask spread'])
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False);(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'catalog_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
