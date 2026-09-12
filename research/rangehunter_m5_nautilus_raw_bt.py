from __future__ import annotations

import argparse, json, math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')
P = dict(adx_period=14,max_adx=23.0,bb_period=20,bb_dev=2.0,max_bb_width_atr=2.20,
         ema_period=50,max_ema_slope_atr=0.18,atr_period=14,rsi_period=14,rsi_buy=35.0,
         rsi_sell=65.0,band_touch=0.20,sl_atr=1.25,tp_atr=0.85,be_r=0.55,
         trail_start_r=0.90,trail_atr=0.65,risk_pct=0.50,max_lot=2.0,
         max_spread_price=1.20,start_hour=7,end_hour=22,friday_stop_hour=20)

class Config(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType

class RangeHunter(Strategy):
    def __init__(self, config: Config):
        super().__init__(config)
        self.bars=deque(maxlen=300); self.armed=None; self.entry=None; self.initial_risk=None
        self.stop=None; self.tp=None; self.middle=None; self.side=None; self.exit_pending=False; self.entries=0
    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def _arr(self,key): return np.array([b[key] for b in self.bars],float)
    def _ema_series(self,x,n):
        if len(x)<n: return None
        alpha=2/(n+1); v=float(np.mean(x[:n])); out=[v]
        for z in x[n:]: v=alpha*float(z)+(1-alpha)*v; out.append(v)
        return out
    def _atr(self,n=14):
        if len(self.bars)<n+1:return None
        xs=list(self.bars); trs=[]
        for i in range(1,len(xs)):
            trs.append(max(xs[i]['h']-xs[i]['l'],abs(xs[i]['h']-xs[i-1]['c']),abs(xs[i]['l']-xs[i-1]['c'])))
        if len(trs)<n:return None
        a=sum(trs[:n])/n
        for tr in trs[n:]: a=(a*(n-1)+tr)/n
        return float(a)
    def _rsi(self,n=14):
        c=self._arr('c')
        if len(c)<n+1:return None
        d=np.diff(c); g=np.maximum(d,0); l=np.maximum(-d,0)
        ag=float(np.mean(g[:n])); al=float(np.mean(l[:n]))
        for i in range(n,len(d)):
            ag=(ag*(n-1)+g[i])/n; al=(al*(n-1)+l[i])/n
        if al==0:return 100.0
        rs=ag/al; return 100-100/(1+rs)
    def _adx(self,n=14):
        xs=list(self.bars)
        if len(xs)<2*n+2:return None
        tr=[]; pdm=[]; mdm=[]
        for i in range(1,len(xs)):
            up=xs[i]['h']-xs[i-1]['h']; dn=xs[i-1]['l']-xs[i]['l']
            pdm.append(up if up>dn and up>0 else 0.0); mdm.append(dn if dn>up and dn>0 else 0.0)
            tr.append(max(xs[i]['h']-xs[i]['l'],abs(xs[i]['h']-xs[i-1]['c']),abs(xs[i]['l']-xs[i-1]['c'])))
        atr=sum(tr[:n]); ps=sum(pdm[:n]); ms=sum(mdm[:n]); dx=[]
        for i in range(n,len(tr)):
            if i>n:
                atr=atr-atr/n+tr[i]; ps=ps-ps/n+pdm[i]; ms=ms-ms/n+mdm[i]
            pdi=100*ps/atr if atr else 0; mdi=100*ms/atr if atr else 0
            den=pdi+mdi; dx.append(100*abs(pdi-mdi)/den if den else 0)
        if len(dx)<n:return None
        a=sum(dx[:n])/n
        for z in dx[n:]: a=(a*(n-1)+z)/n
        return float(a)
    def _snapshot(self):
        if len(self.bars)<60:return None
        c=self._arr('c'); atr=self._atr(); rsi=self._rsi(); adx=self._adx()
        if atr is None or rsi is None or adx is None or atr<=0:return None
        w=c[-P['bb_period']:]; mid=float(np.mean(w)); sd=float(np.std(w,ddof=0)); up=mid+P['bb_dev']*sd; lo=mid-P['bb_dev']*sd
        em=self._ema_series(c,P['ema_period'])
        if em is None or len(em)<2:return None
        return dict(atr=atr,rsi=rsi,adx=adx,mid=mid,up=up,lo=lo,ema=em[-1],ema_prev=em[-2])
    def on_bar(self,bar:Bar):
        b={'o':self.f(bar.open),'h':self.f(bar.high),'l':self.f(bar.low),'c':self.f(bar.close),'ts':int(bar.ts_event)}; self.bars.append(b)
        s=self._snapshot(); self.middle=s['mid'] if s else self.middle
        if self.entry is not None or s is None:return
        width=s['up']-s['lo']
        if s['adx']>P['max_adx'] or width<=0 or width>s['atr']*P['max_bb_width_atr']:return
        if abs(s['ema']-s['ema_prev'])>s['atr']*P['max_ema_slope_atr']:return
        buyzone=b['l']<=s['lo']+width*P['band_touch']; sellzone=b['h']>=s['up']-width*P['band_touch']
        buy=buyzone and s['rsi']<=P['rsi_buy'] and b['c']>b['o'] and b['c']>s['lo']
        sell=sellzone and s['rsi']>=P['rsi_sell'] and b['c']<b['o'] and b['c']<s['up']
        if buy:self.armed=('BUY',s)
        elif sell:self.armed=('SELL',s)
    def on_quote_tick(self,tick:QuoteTick):
        bid=self.f(tick.bid_price); ask=self.f(tick.ask_price); spread=ask-bid
        ts=pd.Timestamp(int(tick.ts_event),unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed is not None and self.entry is None and flat:
            if spread>P['max_spread_price'] or not(P['start_hour']<=ts.hour<P['end_hour']) or (ts.weekday()==4 and ts.hour>=P['friday_stop_hour']): return
            side,s=self.armed; instr=self.cache.instrument(self.config.instrument_id)
            px=ask if side=='BUY' else bid; risk_dist=P['sl_atr']*s['atr']; risk_money=1000.0*P['risk_pct']/100
            qty=min(P['max_lot']*100.0, risk_money/risk_dist if risk_dist>0 else 0)
            if qty<=0:return
            if side=='BUY':
                tp=min(px+P['tp_atr']*s['atr'],s['mid']); stop=px-risk_dist; oside=OrderSide.BUY
                if tp<=px:self.armed=None; return
            else:
                tp=max(px-P['tp_atr']*s['atr'],s['mid']); stop=px+risk_dist; oside=OrderSide.SELL
                if tp>=px:self.armed=None; return
            order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=oside,quantity=instr.make_qty(Decimal(str(qty))))
            self.submit_order(order); self.entry=px; self.initial_risk=risk_dist; self.stop=stop; self.tp=tp; self.middle=s['mid']; self.side=side; self.exit_pending=False; self.entries+=1; self.armed=None; return
        if self.entry is None or self.exit_pending:return
        px=bid if self.side=='BUY' else ask; profit=(px-self.entry) if self.side=='BUY' else (self.entry-px); R=profit/self.initial_risk if self.initial_risk else 0
        s=self._snapshot(); atr=s['atr'] if s else self.initial_risk/P['sl_atr']; mid=s['mid'] if s else self.middle
        if R>0.15 and ((self.side=='BUY' and px>=mid) or (self.side=='SELL' and px<=mid)):
            self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
        if R>=P['be_r']:
            be=self.entry+(0.02 if self.side=='BUY' else -0.02)
            self.stop=max(self.stop,be) if self.side=='BUY' else min(self.stop,be)
        if R>=P['trail_start_r']:
            ns=px-atr*P['trail_atr'] if self.side=='BUY' else px+atr*P['trail_atr']
            self.stop=max(self.stop,ns) if self.side=='BUY' else min(self.stop,ns)
        hit=(px<=self.stop or px>=self.tp) if self.side=='BUY' else (px>=self.stop or px<=self.tp)
        if hit:self.close_all_positions(self.config.instrument_id); self.exit_pending=True
    def on_position_closed(self,event):
        self.entry=self.initial_risk=self.stop=self.tp=self.middle=self.side=None; self.exit_pending=False
    def on_stop(self): self.close_all_positions(self.config.instrument_id)

def parse_money(v):
    if v is None:return 0.0
    try:return float(str(v).replace(',','').split()[0])
    except:return 0.0

def extract(report):
    if report is None or report.empty:return []
    pc=next((c for c in report.columns if 'pnl' in str(c).lower()),None); tc=next((c for c in report.columns if 'closed' in str(c).lower()),None); out=[]
    for i,r in report.iterrows():
        pnl=parse_money(r[pc]) if pc else 0.0; ts=r[tc] if tc else i
        try: t=int(pd.Timestamp(ts).value)
        except: t=i
        out.append({'symbol':'XAUUSD','tf':'M5','pnl':pnl,'ts_closed':t})
    return out

def metrics(trades,initial=1000.0,days=30):
    a=np.array([x['pnl'] for x in trades],float)
    if len(a)==0:return {'N':0,'WR_pct':0,'PF':0,'NetProfit':0,'MaxDD_pct':0,'RF':None,'Monthly21_pct':0,'Daily_pct':0}
    w=a[a>0].sum(); l=abs(a[a<0].sum()); pf=float(w/l) if l else None; eq=initial; peak=initial; mdd=0
    for x in a: eq+=x; peak=max(peak,eq); mdd=max(mdd,peak-eq)
    net=float(a.sum()); dd=100*mdd/peak if peak else 0; m21=((max(eq,1e-9)/initial)**(21/days)-1)*100; daily=((1+m21/100)**(1/21)-1)*100
    return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100),'PF':pf,'NetProfit':net,'MaxDD_pct':dd,'RF':net/mdd if mdd else None,'Monthly21_pct':m21,'Daily_pct':daily}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true'); args=ap.parse_args()
    if not args.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); days=int(manifest['days']); cat=ParquetDataCatalog(str(cp))
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:
        raise SystemExit('XAUUSD missing')
    ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:
        raise SystemExit('no XAUUSD QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks); bt=BarType.from_str(f'{inst.id.value}-5-MINUTE-BID-INTERNAL'); st=RangeHunter(Config(instrument_id=inst.id,bar_type=bt)); eng.add_strategy(st); eng.run()
    tr=extract(eng.trader.generate_positions_report()); met=metrics(tr,days=days); out=Path('results/ae-bt')/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'strategy':'XAUUSD_RangeHunter_M5 parity v1','data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL 5-MINUTE BID bars','execution':'market orders on raw QuoteTicks; observed spread native','period':{'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},'raw_tick_count':len(ticks),'signals_submitted':st.entries,'metrics':met,'limitations':['MT5 broker server timezone mapped to UTC for session gate.','Broker-specific tick value/lot mapping represented as XAU quantity risk sizing: 0.5% initial-equity risk, max 2 lots=200 oz.','Daily-loss and consecutive-loss cooldown account guards are not modeled in this first parity gate; core signal/SL/TP/BE/trail logic is modeled.']}
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False); (out/'summary.json').write_text(json.dumps(summary,indent=2)); (out/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(summary,indent=2)); eng.dispose()
if __name__=='__main__':main()
