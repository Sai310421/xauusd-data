from __future__ import annotations

import argparse
import json
import math
from collections import deque
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LogLevel
from nautilus_trader.config import BacktestEngineConfig, LoggerConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')
TF_MIN = {'M1':1,'M5':5,'M15':15,'H1':60,'H4':240,'D1':1440}
LINEAR = {
    'OTE':[0.62,0.705,0.79],
    'POP':[0.50,0.559,0.669,0.786],
    'GOLD_SILVER':[0.232,0.25,0.688,0.718,0.786,0.804,0.822],
    'ORDER_FLOW':[0.0,0.25,0.5,0.75,1.0],
    'CRT':[-0.40,-0.29,-0.255,-0.21,0.0,1.0,1.47,1.55,2.56,2.60,2.64],
    'SNR':[-0.29,-0.255,-0.21,0.0,1.0,1.55,2.47,2.56,2.60,2.64],
    'STDDEV':[-4,-3,-2,-1,0,1,2,3,4],
    'TARGET':[0.0,0.5,1.0,-1.0,-2.0,-2.5,-3.0,-4.0],
    'MONKEY':[0.63,0.78,0.99,-0.33,-0.66,-0.99],
    'HARMONIC_LIQ':[0.654,0.667,0.697,0.706,0.825,0.835],
}
CIRCLES = {
    'CLASSIC':[0.236,0.382,0.5,0.618,0.786,1.0,1.618],
    'GOLDEN':[0.618,1.618],
    'TIME':[0.236,0.382,0.5,0.618,1.0,1.618,2.618,4.236],
    'HARMONIC':[0.382,0.5,0.618,0.707,0.786,1.0,1.272,1.618,2.618,2.886],
    'LIQUIDITY':[0.5,0.618,0.786,1.0,1.618],
    'MSNR':[1.2,4.5,4.83],
}

class AmosFibCircleConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    trade_size: Decimal
    bar_type_m1: BarType
    bar_type_m5: BarType
    bar_type_m15: BarType
    bar_type_h1: BarType
    bar_type_h4: BarType
    bar_type_d1: BarType
    entry_threshold: float = 0.64


def clamp01(x: float) -> float:
    return max(0.0,min(1.0,float(x)))


def near_score(x: float, levels, tol: float) -> float:
    if not levels: return 0.0
    d=min(abs(x-r) for r in levels)
    return clamp01(1.0-d/max(tol,1e-9))


class AmosFibCircleStrategy(Strategy):
    def __init__(self, config: AmosFibCircleConfig):
        super().__init__(config)
        self.bt = {
            'M1':config.bar_type_m1,'M5':config.bar_type_m5,'M15':config.bar_type_m15,
            'H1':config.bar_type_h1,'H4':config.bar_type_h4,'D1':config.bar_type_d1,
        }
        self.tf_by_bt={str(v):k for k,v in self.bt.items()}
        self.bars={k:deque(maxlen=180) for k in self.bt}
        self.swings={k:deque(maxlen=20) for k in self.bt}
        self.signals={}
        self.mid_moves=deque(maxlen=256)
        self.last_mid=None
        self.entry_ref=None
        self.stop_ref=None
        self.tp_ref=None
        self.side=0
        self.exit_pending=False
        self.entries=0
        self.signal_counts={k:0 for k in self.bt}
        self.boost_entries=0

    @staticmethod
    def _f(px):
        return float(px.as_double()) if hasattr(px,'as_double') else float(px)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        for bt in self.bt.values(): self.subscribe_bars(bt)

    def _atr14(self, tf):
        xs=list(self.bars[tf])
        if len(xs)<15:return None
        trs=[]
        for i in range(-14,0):
            c,p=xs[i],xs[i-1]
            trs.append(max(c['h']-c['l'],abs(c['h']-p['c']),abs(c['l']-p['c'])))
        a=float(np.mean(trs))
        return a if math.isfinite(a) and a>0 else None

    def _maybe_swing(self, tf):
        xs=list(self.bars[tf])
        if len(xs)<5:return
        i=len(xs)-3
        a,b,c=xs[-4],xs[-3],xs[-2]
        # confirmed 3-point swing on closed middle bar
        s=None
        if b['h']>a['h'] and b['h']>=c['h']: s=('H',b['h'],b['ts'])
        elif b['l']<a['l'] and b['l']<=c['l']: s=('L',b['l'],b['ts'])
        if s and (not self.swings[tf] or self.swings[tf][-1][2]!=s[2]): self.swings[tf].append(s)

    def _dow(self,tf):
        sw=list(self.swings[tf]); hs=[x for x in sw if x[0]=='H']; ls=[x for x in sw if x[0]=='L']
        if len(hs)<2 or len(ls)<2:return 0,0.25
        hh=hs[-1][1]>hs[-2][1]; hl=ls[-1][1]>ls[-2][1]
        lh=hs[-1][1]<hs[-2][1]; ll=ls[-1][1]<ls[-2][1]
        if hh and hl:return 1,1.0
        if lh and ll:return -1,1.0
        return 0,0.40

    def _harmonic(self,tf):
        sw=list(self.swings[tf])
        if len(sw)<5:return 0.0
        p=sw[-5:]
        if any(p[i][0]==p[i+1][0] for i in range(4)):return 0.0
        x,a,b,c,d=[q[1] for q in p]
        xa=a-x; ab=b-a; bc=c-b; cd=d-c; xd=d-x
        def rr(u,v): return abs(u)/max(abs(v),1e-9)
        rab,rbc,rcd,rxd=rr(ab,xa),rr(bc,ab),rr(cd,bc),rr(xd,xa)
        s1=near_score(rab,[0.618,0.667,0.706,0.786],0.14)
        s2=near_score(rbc,[0.382,0.5,0.618,0.786],0.16)
        s3=near_score(rcd,[1.272,1.618,2.0,2.618],0.35)
        s4=max(near_score(rxd,[0.697,0.706],0.05),near_score(rxd,[0.825,0.835],0.06),near_score(rxd,[1.272,1.618],0.18))
        return clamp01(0.20*s1+0.15*s2+0.25*s3+0.40*s4)

    def _of_score_signed(self):
        if len(self.mid_moves)<20:return 0.0
        a=np.array(self.mid_moves,float)
        s=np.sign(a[-64:]).sum()/max(len(a[-64:]),1)
        v=a[-64:].sum()/(np.abs(a[-64:]).sum()+1e-9)
        return max(-1.0,min(1.0,0.45*s+0.55*v))

    def _eval(self,tf):
        xs=list(self.bars[tf])
        if len(xs)<30:return
        self._maybe_swing(tf)
        bias,dow=self._dow(tf)
        w=xs[-min(120,len(xs)):]
        lo=min(x['l'] for x in w); hi=max(x['h'] for x in w); span=max(hi-lo,1e-9); close=xs[-1]['c']
        z=(close-lo)/span
        fib=max(near_score(z,levels,0.035 if name not in ('CRT','SNR','STDDEV','TARGET') else 0.12) for name,levels in LINEAR.items())
        sw=list(self.swings[tf]); last_h=next((x[1] for x in reversed(sw) if x[0]=='H'),None); last_l=next((x[1] for x in reversed(sw) if x[0]=='L'),None)
        cur=xs[-1]
        bsl=last_h is not None and cur['h']>last_h and cur['c']<last_h
        ssl=last_l is not None and cur['l']<last_l and cur['c']>last_l
        ict=0.0
        if ssl and bias>=0:ict+=0.65
        if bsl and bias<=0:ict+=0.65
        if z<0.5 and bias==1:ict+=0.25
        if z>0.5 and bias==-1:ict+=0.25
        ict=clamp01(ict)
        anchor=w[0]; dt=max(cur['ts']-anchor['ts'],1); dts=max(w[-1]['ts']-anchor['ts'],1)
        rho=math.sqrt(((close-anchor['c'])/span)**2+(dt/dts)**2)
        circle=max(near_score(rho,levels,0.10) for levels in CIRCLES.values())
        harmonic=self._harmonic(tf)
        ofs=self._of_score_signed(); of=abs(ofs)
        # order-flow proxy may override neutral dow only when fib/ict geometry supports it
        eff_bias=bias
        if eff_bias==0 and ict>=0.55 and fib>=0.45:
            eff_bias=1 if ssl else (-1 if bsl else (1 if ofs>0.35 else (-1 if ofs<-0.35 else 0)))
        total=clamp01(0.18*dow+0.20*ict+0.16*of+0.16*fib+0.13*circle+0.17*harmonic)
        self.signals[tf]={'bias':eff_bias,'score':total,'dow':dow,'ict':ict,'of':of,'fib':fib,'circle':circle,'harmonic':harmonic,'z':z,'rho':rho}
        if total>=self.config.entry_threshold:self.signal_counts[tf]+=1

    def on_bar(self,bar:Bar):
        tf=self.tf_by_bt.get(str(bar.bar_type))
        if tf is None:return
        self.bars[tf].append({'o':self._f(bar.open),'h':self._f(bar.high),'l':self._f(bar.low),'c':self._f(bar.close),'ts':int(bar.ts_event)})
        self._eval(tf)

    def _mtf_decision(self):
        active=[(tf,s) for tf,s in self.signals.items() if s['bias']!=0 and s['score']>=self.config.entry_threshold]
        if not active:return 0,0.0,None,0
        bull=[x for x in active if x[1]['bias']==1]; bear=[x for x in active if x[1]['bias']==-1]
        side=1 if len(bull)>len(bear) else (-1 if len(bear)>len(bull) else 0)
        chosen=bull if side==1 else bear
        if side==0 or not chosen:return 0,0.0,None,0
        # require two aligned TFs, unless strongest single TF is exceptional harmonic/liquidity confluence
        strongest=max(chosen,key=lambda x:x[1]['score'])
        aligned=len(chosen)
        mean=float(np.mean([x[1]['score'] for x in chosen]))
        if aligned<2 and not (strongest[1]['score']>=0.82 and strongest[1]['harmonic']>=0.70):return 0,mean,strongest[0],aligned
        boost=min(1.0,mean+0.05*max(0,aligned-1))
        return side,boost,strongest[0],aligned

    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); mid=(bid+ask)/2
        if self.last_mid is not None:self.mid_moves.append(mid-self.last_mid)
        self.last_mid=mid
        if self.entry_ref is None and self.portfolio.is_net_flat(self.config.instrument_id):
            side,score,tf,aligned=self._mtf_decision()
            if side and score>=self.config.entry_threshold:
                atr=self._atr14(tf)
                if atr is None:return
                inst=self.cache.instrument(self.config.instrument_id)
                qty=inst.make_qty(self.config.trade_size)
                order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side==1 else OrderSide.SELL,quantity=qty)
                self.submit_order(order)
                self.side=side; self.entry_ref=ask if side==1 else bid
                self.stop_ref=self.entry_ref-side*1.25*atr
                self.tp_ref=self.entry_ref+side*2.00*atr
                self.exit_pending=False; self.entries+=1
                if aligned>=3:self.boost_entries+=1
                return
        if self.entry_ref is None or self.exit_pending:return
        px=bid if self.side==1 else ask
        hit_stop=px<=self.stop_ref if self.side==1 else px>=self.stop_ref
        hit_tp=px>=self.tp_ref if self.side==1 else px<=self.tp_ref
        if hit_stop or hit_tp:
            self.close_all_positions(self.config.instrument_id); self.exit_pending=True

    def on_position_closed(self,event):
        self.entry_ref=self.stop_ref=self.tp_ref=None; self.side=0; self.exit_pending=False

    def on_stop(self): self.close_all_positions(self.config.instrument_id)


def parse_money(v):
    if v is None:return 0.0
    if isinstance(v,(float,int,np.number)):return float(v)
    try:return float(str(v).replace(',','').split()[0])
    except Exception:return 0.0


def extract_trades(report):
    if report is None or report.empty:return []
    pnl_col=next((c for c in report.columns if 'pnl' in str(c).lower()),None)
    out=[]
    for i,row in report.iterrows():out.append({'pnl':parse_money(row[pnl_col]) if pnl_col is not None else 0.0,'seq':len(out)})
    return out


def metrics(trades,initial=1000.0,days=90):
    if not trades:return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0}
    a=np.array([x['pnl'] for x in trades],float); wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    eq=initial;peak=initial;mdd=0.0
    for x in a:eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
    net=float(a.sum());ddp=mdd/peak*100 if peak>0 else 0.0
    monthly=((max(eq,1e-9)/initial)**(21/max(days,1))-1)*100
    return {'N':len(a),'WR_pct':float((a>0).mean()*100),'PF':pf,'NetProfit':net,'MaxDD_pct':ddp,'RF':net/mdd if mdd>0 else None,'Monthly21_pct':monthly,'FinalBalance':eq}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');ap.add_argument('--symbol',default='XAUUSD');args=ap.parse_args()
    if not args.raw_bidask_only:raise SystemExit('raw-bidask-only is mandatory')
    cp=Path(args.catalog);manifest=json.loads((cp/'catalog_manifest.json').read_text());days=int(manifest['days']);cat=ParquetDataCatalog(str(cp))
    inst_by={x.id.symbol.value.replace('/',''):x for x in cat.instruments()};inst=inst_by.get(args.symbol)
    if inst is None:raise SystemExit(f'missing instrument {args.symbol}')
    ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:raise SystemExit('no raw QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(trader_id='BT-AMOS-FIB-MTF',logging=LoggerConfig(stdout_level=LogLevel.ERROR),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks)
    bts={tf:BarType.from_str(f'{inst.id.value}-{m}-MINUTE-BID-INTERNAL') for tf,m in TF_MIN.items()}
    strat=AmosFibCircleStrategy(AmosFibCircleConfig(instrument_id=inst.id,trade_size=Decimal('1'),bar_type_m1=bts['M1'],bar_type_m5=bts['M5'],bar_type_m15=bts['M15'],bar_type_h1=bts['H1'],bar_type_h4=bts['H4'],bar_type_d1=bts['D1']))
    eng.add_strategy(strat);eng.run();report=eng.generate_positions_report();trades=extract_trades(report);m=metrics(trades,days=days)
    out=Path('results/ae-bt')/args.experiment_id;out.mkdir(parents=True,exist_ok=True)
    summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','strategy':'AMOS_Dow_ICT_OrderFlow_FibCircle_MTF_v1','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars built directly from raw QuoteTicks','symbol':args.symbol,'timeframes':list(TF_MIN),'period':{'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},'metrics':m,'raw_ticks':len(ticks),'orders_submitted':strat.entries,'boost_entries_3tf_plus':strat.boost_entries,'signal_counts':strat.signal_counts,'last_tf_signals':strat.signals,'limitations':['Order Flow uses a quote-mid directional pressure proxy because the catalog is QuoteTick, not exchange trade aggressor data.','Spread is native via Bid/Ask; explicit commission and probabilistic slippage are not yet added in this first gate.','Circle Fib is implemented as normalized Price-Time radius, not TradingView pixel geometry.','First gate tests one synchronized XAUUSD MTF engine over M1/M5/M15/H1/H4/D1.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));pd.DataFrame(trades).to_csv(out/'trades.csv',index=False);(out/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary,indent=2,ensure_ascii=False));eng.dispose()

if __name__=='__main__':main()
