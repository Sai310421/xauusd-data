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
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import Config,extract,metrics
from research.rangehunter_m1_trendfollow_v2_1_nautilus_raw_bt import ensure_executable_l1,split_503020_integer
from research.rangehunter_m1_trendfollow_v4_nautilus_raw_bt import Strat as V4Strat,P4

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
    def _query_quote_ticks(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_query_quote_ticks

P5=dict(P4)
P5.update(add2_r=0.15,add3_r=0.35,fast_fail_r=-0.45,fast_fail_sec=30)

class Strat(V4Strat):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.pending_legs=None;self.add2_done=False;self.add3_done=False
        self.add2_count=0;self.add3_count=0;self.fast_fail_count=0
    def _submit_units(self,side,units):
        instr=self.cache.instrument(self.config.instrument_id)
        os=OrderSide.BUY if side=='BUY' else OrderSide.SELL
        self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=instr.make_qty(Decimal(units))))
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);dt=pd.Timestamp(ts,unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed and self.entry is None and flat and ts>=self.cool_until:
            side,s=self.armed;spread=ask-bid
            if ts-s['signal_ts']>60_000_000_000:self.armed=None;return
            if spread>P5['max_spread'] or spread>P5['max_spread_atr']*s['atr'] or not(P5['start_hour']<=dt.hour<P5['end_hour']) or (dt.weekday()==4 and dt.hour>=P5['friday_stop_hour']):return
            if not self._tick_accel(side,s['atr'],ts,bid,ask):return
            self.tick_pass+=1;px=ask if side=='BUY' else bid;r=P5['sl_atr']*s['atr'];money=1000*P5['risk_pct']/100;theoretical=min(P5['max_lot']*100,money/r if r>0 else 0);legs=split_503020_integer(int(theoretical))
            if legs is None:self.skipped_small_size+=1;self.armed=None;return
            # V5 DD control: enter only the 50% leg initially. Add 30% and 20% only after favorable continuation.
            self._submit_units(side,legs[0])
            self.pending_legs=legs;self.add2_done=False;self.add3_done=False
            self.entry=px;self.risk=r;self.stop_ref=px-r if side=='BUY' else px+r;self.tp=px+P5['tp_r']*r if side=='BUY' else px-P5['tp_r']*r
            self.side=side;self.entry_ts=ts;self.exit_pending=False;self.baskets+=1;self.armed=None;return
        if self.entry is None or self.exit_pending:return
        px=bid if self.side=='BUY' else ask;profit=px-self.entry if self.side=='BUY' else self.entry-px;R=profit/self.risk if self.risk else 0;a=self.atr(P5['atr']) or self.risk/P5['sl_atr']
        # Favorable-only staged 50/30/20 adds: losers never receive more exposure.
        if self.pending_legs is not None:
            if not self.add2_done and R>=P5['add2_r']:
                self._submit_units(self.side,self.pending_legs[1]);self.add2_done=True;self.add2_count+=1
            if not self.add3_done and R>=P5['add3_r']:
                self._submit_units(self.side,self.pending_legs[2]);self.add3_done=True;self.add3_count+=1
        # Fast-fail is risk control, not an entry filter: exit failed impulse early while keeping N unchanged.
        if (ts-self.entry_ts)<=P5['fast_fail_sec']*1_000_000_000 and R<=P5['fast_fail_r']:
            self.close_all_positions(self.config.instrument_id);self.exit_pending=True;self.fast_fail_count+=1;return
        if R>=P5['be_r']:
            be=self.entry+(0.02 if self.side=='BUY' else -0.02);self.stop_ref=max(self.stop_ref,be) if self.side=='BUY' else min(self.stop_ref,be)
        if R>=P5['trail_start_r']:
            ns=px-P5['trail_atr']*a if self.side=='BUY' else px+P5['trail_atr']*a;self.stop_ref=max(self.stop_ref,ns) if self.side=='BUY' else min(self.stop_ref,ns)
        hit=(px<=self.stop_ref or px>=self.tp) if self.side=='BUY' else (px>=self.stop_ref or px<=self.tp);timeout=(ts-self.entry_ts)>=P5['max_hold_sec']*1_000_000_000
        if hit or timeout:self.close_all_positions(self.config.instrument_id);self.exit_pending=True
    def on_position_closed(self,e):
        super().on_position_closed(e)
        self.pending_legs=None;self.add2_done=False;self.add3_done=False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks,replaced=ensure_executable_l1(raw,1000)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run();pos=eng.trader.generate_positions_report();tr=extract(pos);met=metrics(tr,days=int(man['days']));orders=eng.trader.generate_orders_report();fills=eng.trader.generate_order_fills_report();status={str(k):int(v) for k,v in orders['status'].astype(str).value_counts().to_dict().items()} if orders is not None and not orders.empty and 'status' in orders.columns else {}
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);summary=dict(verification_level='NAUTILUS_BT_RAW_BIDASK_L1_SIZE_ASSUMPTION',engine='NautilusTrader BacktestEngine',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),strategy='RangeHunter_M1_TrendFollow_v5_Staged503020',data_kind='RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes',ohlc_resample_used=False,logic='V4 AdaptiveMTF signal + staged 50/30/20 favorable-only adds + fast-fail DD control',period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),raw_tick_count=len(raw),l1_size_replacements=replaced,diagnostics=dict(bar_setups=st.setup_count,m5_bias_pass=st.bias_pass,sweep_pass=st.sweep_pass,bos_pass=st.bos_pass,tick_accel_pass=st.tick_pass,baskets_submitted=st.baskets,add2_count=st.add2_count,add3_count=st.add3_count,fast_fail_count=st.fast_fail_count,skipped_small_size=st.skipped_small_size,order_status=status,fill_rows=0 if fills is None else len(fills),position_rows=0 if pos is None else len(pos)),params=P5,metrics=met,limitations=['Raw Bid/Ask prices/timestamps unchanged; zero quote sizes replaced by synthetic L1 execution size 1000.','M5 bias is built from sequential M1 INTERNAL bars sourced from the same raw ticks.','UTC session mapping.','No explicit commission/probabilistic slippage beyond native spread.'])
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False);orders.to_csv(out/'orders.csv',index=False) if orders is not None else None;fills.to_csv(out/'fills.csv',index=False) if fills is not None else None;(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'catalog_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
