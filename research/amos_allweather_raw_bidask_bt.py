from __future__ import annotations

import argparse, json, math
from collections import Counter, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
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

SIM=Venue('SIM')
TF_MIN={'M1':1,'M5':5,'M15':15}

class Scene(str,Enum):
    COMPRESSION='compression'; BALANCED_RANGE='balanced_range'; LIQUIDITY_BUILD='liquidity_build'; SWEEP_REJECTION='sweep_rejection'; PRE_BREAKOUT='pre_breakout'; EXPANDING_RANGE='expanding_range'; NOISE='noise'; RETRACEMENT='retracement'; CONTINUATION='continuation'; REVERSAL='reversal'; BREAKOUT='breakout'; GAP='gap'; NEWS='news'; CRISIS='crisis'; TRANSITION='transition'

@dataclass
class F:
    bb_width_pct:float=50.; adx:float=25.; adx_slope:float=0.; atr_pct:float=50.; atr_slope:float=0.; efficiency_ratio:float=.5
    boundary_rejections:int=0; equal_highs:int=0; equal_lows:int=0; sweep:bool=False; sweep_side:str=''; return_inside:bool=False
    cisd:bool=False; internal_mss:bool=False; external_break:bool=False; break_side:str=''; displacement:bool=False; outside_acceptance:bool=False
    fvg:bool=False; ifvg:bool=False; bpr:bool=False; breaker:bool=False; gap_atr:float=0.; news_tier:int=0; spread_z:float=0.; velocity_z:float=0.; slippage_z:float=0.
    eq:float=0.; range_high:float=0.; range_low:float=0.

@dataclass
class Decision:
    scene:Scene; confidence:float; crt_score:float; engines:dict[str,float]; reasons:list[str]=field(default_factory=list)

class Router:
    @staticmethod
    def crt(x:F)->float:
        raw=30*x.sweep+20*x.displacement+20*x.cisd+15*x.breaker+10*x.ifvg+10*x.bpr+5*x.fvg+5*(x.adx_slope>0)+5*(x.atr_slope>0)
        return min(100.,100.*raw/120.)
    def classify(self,x:F)->Decision:
        c=self.crt(x)
        if max(x.spread_z,x.velocity_z,x.slippage_z)>=5:return Decision(Scene.CRISIS,.95,c,{'crisis':.8,'reserve':.2},['shock'])
        if x.news_tier>=2:return Decision(Scene.NEWS,.9,c,{'news':.6,'reserve':.4},['scheduled event'])
        if x.gap_atr>=.8:return Decision(Scene.GAP,.85,c,{'gap':.7,'reserve':.3},['gap'])
        if x.sweep and x.return_inside and x.cisd and x.internal_mss and x.displacement:
            if x.external_break and c>=75:return Decision(Scene.REVERSAL,min(.99,c/100),c,{'reversal_rr':.8,'reserve':.2},['CRT confirmed'])
            return Decision(Scene.SWEEP_REJECTION,max(.6,c/100),c,{'cb':.35,'crt_probe':.45,'reserve':.2},['sweep rejected'])
        if x.external_break and x.displacement and x.outside_acceptance:return Decision(Scene.BREAKOUT,.9,c,{'expansion_rr':.8,'reserve':.2},['outside acceptance'])
        if x.bb_width_pct<=20 and x.adx<20 and x.atr_pct<=25:
            if x.adx_slope>0 and x.atr_slope>0:return Decision(Scene.PRE_BREAKOUT,.8,c,{'cb':.2,'breakout_probe':.6,'reserve':.2},['compression release'])
            if x.efficiency_ratio<.30 and x.boundary_rejections>=2:
                if x.equal_highs>=2 or x.equal_lows>=2:return Decision(Scene.LIQUIDITY_BUILD,.85,c,{'cb':.7,'crt_watch':.1,'reserve':.2},['liquidity build'])
                return Decision(Scene.BALANCED_RANGE,.85,c,{'cb':.8,'reserve':.2},['balanced range'])
            return Decision(Scene.COMPRESSION,.7,c,{'compression':.5,'reserve':.5},['range candidate'])
        if x.adx<20 and x.atr_pct>=70:return Decision(Scene.EXPANDING_RANGE,.7,c,{'volatility':.5,'reserve':.5},['high vol no direction'])
        if x.external_break and not x.outside_acceptance:return Decision(Scene.RETRACEMENT,.65,c,{'pullback':.6,'reserve':.4},['break not accepted'])
        if x.displacement and x.adx>=20:return Decision(Scene.CONTINUATION,.75,c,{'trend':.7,'reserve':.3},['directional expansion'])
        return Decision(Scene.TRANSITION,.5,c,{'discovery':.2,'reserve':.8},['insufficient edge'])

class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId; bar_type:BarType; trade_size:Decimal; tf_minutes:int

class Strat(Strategy):
    def __init__(self,cfg:Cfg):
        super().__init__(cfg); self.bars=deque(maxlen=160); self.router=Router(); self.decision=Decision(Scene.TRANSITION,0,0,{'reserve':1})
        self.bias=0; self.entry_ref=None; self.stop_ref=None; self.tp_ref=None; self.trail_ref=None; self.entry_side=0; self.hold=0; self.exit_pending=False; self.entries=0
        self.entry_scenes=[]; self.scene_counts=Counter(); self.transitions=Counter(); self.prev_scene=Scene.TRANSITION; self.spreads=deque(maxlen=3000); self.rets=deque(maxlen=3000); self.last_mid=None; self.last_bid=None; self.last_ask=None
    @staticmethod
    def f(px):return float(px.as_double()) if hasattr(px,'as_double') else float(px)
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
    @staticmethod
    def pct(vals,x):
        if not vals:return 50.
        a=np.asarray(vals,float); return float((a<=x).mean()*100)
    def atrs(self,n=14):
        xs=list(self.bars); tr=[]
        for i in range(1,len(xs)):
            c,p=xs[i],xs[i-1]; tr.append(max(c['h']-c['l'],abs(c['h']-p['c']),abs(c['l']-p['c'])))
        if len(tr)<n:return []
        return [float(np.mean(tr[max(0,i-n+1):i+1])) for i in range(n-1,len(tr))]
    def adx(self,n=14):
        xs=list(self.bars)
        if len(xs)<n+2:return 25.
        tr=[]; pdm=[]; ndm=[]
        for i in range(1,len(xs)):
            c,p=xs[i],xs[i-1]; up=c['h']-p['h']; dn=p['l']-c['l']; pdm.append(up if up>dn and up>0 else 0); ndm.append(dn if dn>up and dn>0 else 0); tr.append(max(c['h']-c['l'],abs(c['h']-p['c']),abs(c['l']-p['c'])))
        s=sum(tr[-n:]);
        if s<=0:return 0.
        pdi=100*sum(pdm[-n:])/s; ndi=100*sum(ndm[-n:])/s; return 100*abs(pdi-ndi)/max(pdi+ndi,1e-9)
    def features(self):
        xs=list(self.bars)
        if len(xs)<40:return None
        cur,prev=xs[-1],xs[-2]; av=self.atrs();
        if not av:return None
        atr=av[-1]
        if atr<=0:return None
        atr_pct=self.pct(av[-80:],atr); atr_slope=atr-(av[-4] if len(av)>=4 else atr)
        closes=np.asarray([b['c'] for b in xs],float); widths=[]
        for i in range(20,len(closes)+1):
            w=closes[i-20:i]; m=float(np.mean(w)); s=float(np.std(w)); widths.append(4*s/m if m else 0)
        bb=self.pct(widths[-80:],widths[-1] if widths else 0); adx=self.adx()
        d1=abs(xs[-1]['c']-xs[-8]['c'])/max(sum(abs(xs[j]['c']-xs[j-1]['c']) for j in range(len(xs)-7,len(xs))),1e-9)
        d0=abs(xs[-5]['c']-xs[-12]['c'])/max(sum(abs(xs[j]['c']-xs[j-1]['c']) for j in range(len(xs)-11,len(xs)-4)),1e-9); adx_slope=d1-d0
        er=abs(xs[-1]['c']-xs[-21]['c'])/max(sum(abs(xs[j]['c']-xs[j-1]['c']) for j in range(len(xs)-20,len(xs))),1e-9)
        lb=xs[-21:-1]; rh=max(b['h'] for b in lb); rl=min(b['l'] for b in lb); eq=(rh+rl)/2; tol=max(.10*atr,1e-9)
        eh=sum(abs(b['h']-rh)<=tol for b in lb); el=sum(abs(b['l']-rl)<=tol for b in lb); br=sum((b['h']>=rh-tol and b['c']<rh-tol/2) or (b['l']<=rl+tol and b['c']>rl+tol/2) for b in lb[-10:])
        us=cur['h']>rh and cur['c']<rh; ls=cur['l']<rl and cur['c']>rl; sweep=us or ls; ss='upper' if us else ('lower' if ls else '')
        eu=cur['c']>rh; ed=cur['c']<rl; eb=eu or ed; bs='up' if eu else ('down' if ed else '')
        rng=max(cur['h']-cur['l'],1e-9); disp=rng>=1.5*atr and abs(cur['c']-cur['o'])/rng>=.60; acc=(eu and prev['c']>rh) or (ed and prev['c']<rl)
        cisd=(us and cur['c']<prev['o']) or (ls and cur['c']>prev['o']); ih=max(b['h'] for b in xs[-7:-1]); il=min(b['l'] for b in xs[-7:-1]); mss=(us and cur['c']<il) or (ls and cur['c']>ih)
        fvg=xs[-1]['l']>xs[-3]['h'] or xs[-1]['h']<xs[-3]['l']; gap=abs(cur['o']-prev['c'])/atr
        sz=0.; vz=0.
        if self.last_bid is not None and len(self.spreads)>=50:
            sp=self.last_ask-self.last_bid; a=np.asarray(self.spreads,float); sd=float(a.std()); sz=(sp-float(a.mean()))/sd if sd>0 else 0
        if len(self.rets)>=50:
            a=np.asarray(self.rets,float); sd=float(a.std()); vz=abs(float(a[-1])-float(a.mean()))/sd if sd>0 else 0
        return F(bb,adx,adx_slope,atr_pct,atr_slope,er,br,eh,el,sweep,ss,sweep,cisd,mss,eb,bs,disp,acc,fvg,False,False,False,gap,0,sz,vz,0,eq,rh,rl)
    def direction(self,x:F,d:Decision):
        c=self.bars[-1]
        if d.scene in (Scene.BALANCED_RANGE,Scene.LIQUIDITY_BUILD,Scene.COMPRESSION):
            loc=(c['c']-x.range_low)/max(x.range_high-x.range_low,1e-9); return -1 if loc>=.70 else (1 if loc<=.30 else 0)
        if d.scene in (Scene.SWEEP_REJECTION,Scene.REVERSAL):return -1 if x.sweep_side=='upper' else (1 if x.sweep_side=='lower' else 0)
        if d.scene in (Scene.BREAKOUT,Scene.CONTINUATION,Scene.PRE_BREAKOUT):return 1 if x.break_side=='up' else (-1 if x.break_side=='down' else (1 if c['c']>c['o'] else -1))
        if d.scene==Scene.RETRACEMENT:return self.bias
        if d.scene in (Scene.EXPANDING_RANGE,Scene.CRISIS):return 1 if c['c']>c['o'] and x.displacement else (-1 if c['c']<c['o'] and x.displacement else 0)
        if d.scene==Scene.GAP:
            gd=1 if c['o']>self.bars[-2]['c'] else -1; return gd if x.outside_acceptance else -gd
        return 0
    def on_bar(self,bar:Bar):
        self.bars.append({'o':self.f(bar.open),'h':self.f(bar.high),'l':self.f(bar.low),'c':self.f(bar.close),'ts':int(bar.ts_event)})
        if self.entry_ref is not None:self.hold+=1
        x=self.features()
        if x is None:return
        d=self.router.classify(x); self.decision=d; self.scene_counts[d.scene.value]+=1
        if d.scene!=self.prev_scene:self.transitions[f'{self.prev_scene.value}->{d.scene.value}']+=1; self.prev_scene=d.scene
        if d.scene in (Scene.BREAKOUT,Scene.CONTINUATION) and x.break_side:self.bias=1 if x.break_side=='up' else -1
        if d.scene==Scene.REVERSAL and x.sweep_side:self.bias=-1 if x.sweep_side=='upper' else 1
        if self.entry_ref is not None and self.hold>=max(2,int(180/self.config.tf_minutes)) and not self.exit_pending:self.close_all_positions(self.config.instrument_id); self.exit_pending=True
    def on_quote_tick(self,tick:QuoteTick):
        bid,ask=self.f(tick.bid_price),self.f(tick.ask_price); mid=(bid+ask)/2
        if self.last_mid is not None:self.rets.append(mid-self.last_mid)
        self.last_mid=mid; self.spreads.append(max(ask-bid,0)); self.last_bid,self.last_ask=bid,ask
        x=self.features()
        if x is None:return
        d=self.decision
        if self.entry_ref is None and self.portfolio.is_net_flat(self.config.instrument_id):
            side=self.direction(x,d)
            if side and d.confidence>=.65 and d.scene not in (Scene.TRANSITION,Scene.NOISE,Scene.NEWS):
                ins=self.cache.instrument(self.config.instrument_id); order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=ins.make_qty(self.config.trade_size)); self.submit_order(order)
                self.entry_ref=ask if side>0 else bid; self.entry_side=side; self.entry_scenes.append(d.scene.value); atr=self.atrs()[-1]
                if d.scene in (Scene.BALANCED_RANGE,Scene.LIQUIDITY_BUILD,Scene.COMPRESSION):sk,tk=.75,.55
                elif d.scene in (Scene.REVERSAL,Scene.BREAKOUT,Scene.CONTINUATION,Scene.CRISIS):sk,tk=1.,1.8
                else:sk,tk=.9,1.2
                self.stop_ref=self.entry_ref-side*sk*atr; self.tp_ref=self.entry_ref+side*tk*atr; self.trail_ref=None; self.hold=0; self.exit_pending=False; self.entries+=1; return
        if self.entry_ref is None or self.exit_pending:return
        px=bid if self.entry_side>0 else ask; fav=(px-self.entry_ref)*self.entry_side; risk=abs(self.entry_ref-self.stop_ref)
        if fav>=.8*risk:
            cand=px-self.entry_side*.45*risk; self.trail_ref=cand if self.trail_ref is None else (max(self.trail_ref,cand) if self.entry_side>0 else min(self.trail_ref,cand))
        st=self.stop_ref if self.trail_ref is None else (max(self.stop_ref,self.trail_ref) if self.entry_side>0 else min(self.stop_ref,self.trail_ref)); hit=(px<=st or px>=self.tp_ref) if self.entry_side>0 else (px>=st or px<=self.tp_ref)
        if hit:self.close_all_positions(self.config.instrument_id); self.exit_pending=True
    def on_position_closed(self,event):self.entry_ref=self.stop_ref=self.tp_ref=self.trail_ref=None; self.entry_side=0; self.hold=0; self.exit_pending=False
    def on_stop(self):self.close_all_positions(self.config.instrument_id)

def money(v):
    if v is None:return 0.
    if isinstance(v,(float,int,np.number)):return float(v)
    try:return float(str(v).replace(',','').split()[0])
    except:return 0.

def trades_from(report,symbol,tf,scenes):
    if report is None or report.empty:return []
    pc=next((c for c in report.columns if 'pnl' in str(c).lower()),None); tc=next((c for c in report.columns if 'closed' in str(c).lower() and ('ts' in str(c).lower() or 'time' in str(c).lower())),None); out=[]
    for j,(i,row) in enumerate(report.iterrows()):
        ts=row[tc] if tc is not None else i
        try:ts=pd.Timestamp(ts).value
        except:
            try:ts=int(ts)
            except:ts=j
        out.append({'symbol':symbol,'tf':tf,'scene':scenes[j] if j<len(scenes) else 'unknown','pnl':money(row[pc]) if pc else 0.,'ts_closed':int(ts)})
    return out

def metrics(ts,initial=1000.,days=30):
    if not ts:return {'N':0,'WR_pct':0.,'PF':0.,'NetProfit':0.,'MaxDD_pct':0.,'RF':None,'Monthly21_pct':0.,'Daily_pct':0.}
    a=np.array([t['pnl'] for t in ts],float); w=a[a>0]; l=a[a<0]; pf=float(w.sum()/abs(l.sum())) if len(l) and l.sum()!=0 else (float('inf') if len(w) else 0.); eq=initial; peak=initial; dd=0.
    for p in a:eq+=p; peak=max(peak,eq); dd=max(dd,peak-eq)
    net=float(a.sum()); ddp=dd/peak*100 if peak>0 else 0.; m=((max(eq,1e-9)/initial)**(21/days)-1)*100; daily=((1+m/100)**(1/21)-1)*100 if m>-100 else -100
    return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100),'PF':pf,'NetProfit':net,'MaxDD_pct':float(ddp),'RF':float(net/dd) if dd>0 else None,'Monthly21_pct':float(m),'Daily_pct':float(daily)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--symbols',nargs='+',required=True); ap.add_argument('--timeframes',nargs='+',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true'); a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only is mandatory')
    path=Path(a.catalog); manifest=json.loads((path/'catalog_manifest.json').read_text()); days=int(manifest['days']); cat=ParquetDataCatalog(str(path)); insts={x.id.symbol.value.replace('/',''):x for x in cat.instruments()}; out=Path('results/ae-bt')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    allts=[]; cells={}; scenes={}; counts={}; trans={}; raw={}
    for symbol in [s for s in a.symbols if s=='XAUUSD']:
        ins=insts.get(symbol)
        if ins is None:raise SystemExit(f'instrument missing: {symbol}')
        ticks=cat.query_quote_ticks(identifiers=[ins.id.value]); raw[symbol]=len(ticks)
        if not ticks:raise SystemExit('no raw QuoteTicks: XAUUSD')
        for tf in a.timeframes:
            mins=TF_MIN[tf]; eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True))); eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000')); eng.add_instrument(ins); eng.add_data(ticks)
            bt=BarType.from_str(f'{ins.id.value}-{mins}-MINUTE-BID-INTERNAL'); st=Strat(Cfg(instrument_id=ins.id,bar_type=bt,trade_size=Decimal('1'),tf_minutes=mins)); eng.add_strategy(st); eng.run(); tt=trades_from(eng.generate_positions_report(),symbol,tf,st.entry_scenes); allts+=tt; key=f'{symbol}:{tf}'; cells[key]={**metrics(tt,days=days),'raw_ticks':len(ticks),'signals_submitted':st.entries}; counts[key]=dict(st.scene_counts); trans[key]=dict(st.transitions)
            for sc in sorted({t['scene'] for t in tt}):scenes[f'{key}:{sc}']=metrics([t for t in tt if t['scene']==sc],days=days)
            eng.dispose()
    allts.sort(key=lambda x:(x['ts_closed'],x['symbol'],x['tf'])); summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','strategy':'AMOS_AllWeather_XAUUSD_MetaBot_v0.2','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars built from raw QuoteTicks','execution':'MARKET orders on raw QuoteTicks; native observed spread included; explicit commission/slippage model not yet added','symbols':['XAUUSD'],'timeframes':a.timeframes,'period':{'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},'portfolio_realized_close_ordered':metrics(allts,days=days),'cell_metrics':cells,'scene_metrics':scenes,'scene_bar_counts':counts,'state_transitions':trans,'raw_tick_counts':raw,'limitations':['Scheduled-news calendar is not present in the raw catalog; NEWS is not directly event-labelled in v0.2.','IFVG/BPR/Breaker are reserved but not fully reconstructed in v0.2.','Portfolio DD is reconstructed from realized closed PnL across independent TF cells, not synchronized mark-to-market equity.','No explicit commission/probabilistic slippage model yet; raw Bid/Ask spread is native.']}
    pd.DataFrame(allts).to_csv(out/'trades.csv',index=False); (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); (out/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)); print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
