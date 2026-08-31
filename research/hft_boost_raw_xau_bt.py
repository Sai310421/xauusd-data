from __future__ import annotations
import argparse, json
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')

class HFTBaseConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    trade_size: Decimal
    point: float
    min_velocity: float = 8.0
    min_imbalance: float = 20.0
    min_score: float = 62.0
    tp_points: float = 8.0
    sl_points: float = 10.0
    max_hold_ms: int = 15000
    cooldown_ms: int = 250

class HFTBaseStrategy(Strategy):
    def __init__(self, config: HFTBaseConfig):
        super().__init__(config)
        self.ticks=deque(maxlen=500); self.prev_velocity=0.0
        self.entry_price=None; self.entry_side=None; self.entry_ts_ms=None
        self.exit_pending=False; self.last_exit_ts_ms=-10**18
        self.entries=0; self.signal_count=0; self.score_sum=0.0
    @staticmethod
    def _f(px): return float(px.as_double()) if hasattr(px,'as_double') else float(px)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def _micro(self,tick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); ts_ms=int(tick.ts_event//1_000_000)
        self.ticks.append((ts_ms,bid,ask))
        if len(self.ticks)<30:return None
        items=list(self.ticks); n=min(30,len(items))
        while n<min(len(items),250):
            if (items[-1][0]-items[-n][0])/1000.0>=1.0:break
            n+=1
        sample=items[-n:]; seconds=max((sample[-1][0]-sample[0][0])/1000.0,0.001)
        ch=[sample[i][1]-sample[i-1][1] for i in range(1,len(sample))]
        velocity=(sum(ch)/self.config.point)/seconds; accel=velocity-self.prev_velocity; self.prev_velocity=velocity
        up=sum(x>0 for x in ch); dn=sum(x<0 for x in ch); den=up+dn
        imb=((up-dn)/den*100.0) if den else 0.0; spread=max(0.0,(ask-bid)/self.config.point)
        exhaustion=abs(velocity)>30 and accel*velocity<0
        return ts_ms,bid,ask,velocity,imb,spread,exhaustion
    def _signal(self,m):
        _,_,_,vel,imb,spread,exhaustion=m
        buy=vel>=self.config.min_velocity and imb>=self.config.min_imbalance
        sell=vel<=-self.config.min_velocity and imb<=-self.config.min_imbalance
        if not (buy or sell):return None
        side='buy' if buy else 'sell'; score=min(100.0,50+abs(vel)*0.6+abs(imb)*0.25)
        if exhaustion:score-=20
        if spread<=25:score+=5
        frac=spread/max(self.config.tp_points,1e-9)
        if frac>.25:score-=12
        elif frac>.15:score-=7
        elif frac>.08:score-=3
        self.signal_count+=1; self.score_sum+=score
        return (side,score) if score>=self.config.min_score else None
    def on_quote_tick(self,tick:QuoteTick):
        m=self._micro(tick)
        if m is None:return
        ts_ms,bid,ask,*_=m
        if self.entry_price is not None and not self.exit_pending:
            signed=1 if self.entry_side=='buy' else -1; mark=bid if self.entry_side=='buy' else ask
            dpts=signed*(mark-self.entry_price)/self.config.point; held=ts_ms-self.entry_ts_ms
            if dpts>=self.config.tp_points or dpts<=-self.config.sl_points or held>=self.config.max_hold_ms:
                self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
        if self.entry_price is not None or self.exit_pending:return
        if ts_ms-self.last_exit_ts_ms<self.config.cooldown_ms:return
        s=self._signal(m)
        if s is None:return
        side,_=s; instrument=self.cache.instrument(self.config.instrument_id)
        order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side=='buy' else OrderSide.SELL,quantity=instrument.make_qty(self.config.trade_size))
        self.submit_order(order); self.entry_price=ask if side=='buy' else bid; self.entry_side=side; self.entry_ts_ms=ts_ms; self.entries+=1
    def on_position_closed(self,event):
        try:ts_ms=int(event.ts_event//1_000_000)
        except Exception:ts_ms=self.entry_ts_ms or 0
        self.last_exit_ts_ms=ts_ms; self.entry_price=None; self.entry_side=None; self.entry_ts_ms=None; self.exit_pending=False
    def on_stop(self):
        if self.entry_price is not None:self.close_all_positions(self.config.instrument_id)

def parse_money(v):
    if v is None:return 0.0
    if isinstance(v,(float,int,np.number)):return float(v)
    try:return float(str(v).replace(',','').strip().split()[0])
    except Exception:return 0.0

def extract_trades(report):
    if report is None or report.empty:return []
    pnl_col=next((c for c in report.columns if str(c).lower() in ('realized_pnl','pnl','realizedpnl')),None)
    if pnl_col is None:pnl_col=next((c for c in report.columns if 'pnl' in str(c).lower()),None)
    ts_col=next((c for c in report.columns if 'closed' in str(c).lower() and ('ts' in str(c).lower() or 'time' in str(c).lower())),None)
    out=[]
    for i,row in report.iterrows():
        pnl=parse_money(row[pnl_col]) if pnl_col is not None else 0.0; ts=row[ts_col] if ts_col is not None else i
        try:ts=pd.Timestamp(ts).value
        except Exception:
            try:ts=int(ts)
            except Exception:ts=len(out)
        out.append({'pnl':pnl,'ts_closed':int(ts)})
    return out

def metrics(trades,initial,days):
    a=np.array([t['pnl'] for t in trades],float)
    if len(a)==0:return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxClosedDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0,'Daily_pct':0.0}
    wins=a[a>0]; losses=a[a<0]; pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    eq=initial; peak=initial; mdd=0.0
    for x in a:eq+=x; peak=max(peak,eq); mdd=max(mdd,peak-eq)
    net=float(a.sum()); ddp=mdd/peak*100 if peak>0 else 0.0; monthly=((max(eq,1e-9)/initial)**(21/days)-1)*100
    daily=((1+monthly/100)**(1/21)-1)*100 if monthly>-100 else -100.0
    return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100),'PF':pf,'NetProfit':net,'MaxClosedDD_pct':float(ddp),'RF':float(net/mdd) if mdd>0 else None,'Monthly21_pct':float(monthly),'Daily_pct':float(daily),'FinalBalance':float(eq)}

def install_quote_compat():
    if hasattr(ParquetDataCatalog,'query_quote_ticks'):return
    if hasattr(ParquetDataCatalog,'quotes'):
        def query_quote_ticks(self,identifiers=None,start=None,end=None,**kwargs):return self.quotes(instrument_ids=identifiers,start=start,end=end,**kwargs)
        ParquetDataCatalog.query_quote_ticks=query_quote_ticks; return
    raise RuntimeError('No Raw QuoteTick reader found')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true')
    ap.add_argument('--initial',type=float,default=1000.0); ap.add_argument('--min-score',type=float,default=62.0); ap.add_argument('--tp-points',type=float,default=8.0); ap.add_argument('--sl-points',type=float,default=10.0); ap.add_argument('--trade-size',default='1')
    args=ap.parse_args()
    if not args.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    install_quote_compat(); cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); days=int(manifest['days']); catalog=ParquetDataCatalog(str(cp))
    instrument={x.id.symbol.value.replace('/',''):x for x in catalog.instruments()}.get('XAUUSD')
    if instrument is None:raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks:raise SystemExit('no XAUUSD raw QuoteTicks')
    try:point=float(instrument.price_increment.as_double())
    except Exception:point=float(str(instrument.price_increment))
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(args.initial,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(instrument); eng.add_data(ticks)
    strat=HFTBaseStrategy(HFTBaseConfig(instrument_id=instrument.id,trade_size=Decimal(args.trade_size),point=point,min_score=args.min_score,tp_points=args.tp_points,sl_points=args.sl_points))
    eng.add_strategy(strat); eng.run(); trades=extract_trades(eng.generate_positions_report()); k=metrics(trades,args.initial,days)
    outdir=Path('results/ae-bt')/args.experiment_id; outdir.mkdir(parents=True,exist_ok=True); pd.DataFrame(trades).to_csv(outdir/'trades.csv',index=False)
    summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','edge':'HFT_BOOST_BASE_v0.4','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'execution':'Nautilus native MARKET orders; native Bid/Ask spread; no explicit fee/slippage yet','period':{'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},'raw_ticks':len(ticks),'point':point,'signals':strat.signal_count,'entries_submitted':strat.entries,'avg_signal_score':(strat.score_sum/strat.signal_count if strat.signal_count else 0.0),'params':{'min_score':args.min_score,'tp_points':args.tp_points,'sl_points':args.sl_points,'trade_size':args.trade_size},'kpi':k,'limitations':['Floating mark-to-market DD/MAE/MFE and explicit commission/slippage are next Reality Gate; this run is the Raw BidAsk BASE gate.']}
    text=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True); (outdir/'summary.json').write_text(text); (outdir/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2)); print(text); eng.dispose()
if __name__=='__main__':main()
