from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
import pandas as pd, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import Strat as BaseStrat,Config,P,extract,metrics


def _f(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

def ensure_executable_l1(raw_ticks, depth_units=1000):
    """Preserve every raw bid/ask price and timestamp; replace only zero L1 sizes.
    Dukascopy cache has valid prices but zero quote sizes, which Nautilus treats as no market.
    depth_units is execution-only synthetic L1 liquidity, not price fabrication.
    """
    depth=Quantity.from_int(depth_units); out=[]; replaced=0
    for t in raw_ticks:
        bs=_f(t.bid_size); az=_f(t.ask_size)
        if bs<=0 or az<=0:
            out.append(QuoteTick(
                instrument_id=t.instrument_id,
                bid_price=t.bid_price,
                ask_price=t.ask_price,
                bid_size=depth if bs<=0 else t.bid_size,
                ask_size=depth if az<=0 else t.ask_size,
                ts_event=t.ts_event,
                ts_init=t.ts_init,
            )); replaced+=1
        else: out.append(t)
    return out,replaced

def split_503020_integer(total_units:int):
    """Risk-safe integer allocation. Sum never exceeds floored risk-sized total."""
    if total_units<3:return None
    q1=max(1,int(round(total_units*0.50)));q2=max(1,int(round(total_units*0.30)));q3=total_units-q1-q2
    while q3<1 and q1>1:q1-=1;q3+=1
    while q3<1 and q2>1:q2-=1;q3+=1
    if min(q1,q2,q3)<1:return None
    return (q1,q2,q3)

class Strat(BaseStrat):
    def __init__(self,cfg):
        super().__init__(cfg);self.skipped_small_size=0;self.split_units=[]
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);dt=pd.Timestamp(ts,unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed and self.entry is None and flat and ts>=self.cool_until:
            if ask-bid>P['max_spread'] or not(P['start_hour']<=dt.hour<P['end_hour']) or (dt.weekday()==4 and dt.hour>=P['friday_stop_hour']):return
            side,s=self.armed;px=ask if side=='BUY' else bid;r=P['sl_atr']*s['atr'];money=1000*P['risk_pct']/100
            theoretical=min(P['max_lot']*100,money/r if r>0 else 0);total_units=int(theoretical)
            legs=split_503020_integer(total_units)
            if legs is None:
                self.skipped_small_size+=1;self.armed=None;return
            instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if side=='BUY' else OrderSide.SELL
            for units in legs:
                q=instr.make_qty(Decimal(units));self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=q))
            self.split_units.append(legs)
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks,replaced=ensure_executable_l1(raw,1000)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks);bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run()
    pos=eng.trader.generate_positions_report();tr=extract(pos);met=metrics(tr,days=int(man['days']));orders=eng.trader.generate_orders_report();fills=eng.trader.generate_order_fills_report()
    status={str(k):int(v) for k,v in orders['status'].astype(str).value_counts().to_dict().items()} if orders is not None and not orders.empty and 'status' in orders.columns else {}
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True)
    summary=dict(
      verification_level='NAUTILUS_BT_RAW_BIDASK_L1_SIZE_ASSUMPTION',engine='NautilusTrader BacktestEngine',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),
      strategy='RangeHunter_M1_TrendFollow_v2.1',data_kind='RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes',ohlc_resample_used=False,
      signal_bars='Nautilus INTERNAL 1-MINUTE BID bars',execution='3 market entries, integer approximation to 50/30/20, native raw spread; zero quote sizes replaced by 1000 execution units',
      period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),raw_tick_count=len(raw),l1_size_replacements=replaced,
      diagnostics=dict(bar_setups=st.setup_count,direction_pass=st.direction_count,pullback_pass=st.pullback_count,baskets_submitted=st.baskets,skipped_small_size=st.skipped_small_size,order_status=status,fill_rows=0 if fills is None else len(fills),position_rows=0 if pos is None else len(pos),sample_splits=st.split_units[:20]),
      params=P,metrics=met,
      limitations=['Raw Bid/Ask prices and timestamps are unchanged; quote sizes are synthetic because catalog sizes are zero.','UTC session mapping.','0.5% initial-equity risk; integer oz flooring means realized risk is at or below theoretical risk.','No explicit commission or probabilistic slippage beyond native Bid/Ask spread.'])
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False);orders.to_csv(out/'orders.csv',index=False) if orders is not None else None;fills.to_csv(out/'fills.csv',index=False) if fills is not None else None
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'catalog_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
