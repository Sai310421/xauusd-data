from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
import pandas as pd, numpy as np, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick,Bar
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import Strat as BaseStrat,Config,extract,metrics
from research.rangehunter_m1_trendfollow_v2_1_nautilus_raw_bt import ensure_executable_l1,split_503020_integer

P3=dict(ema=21,atr=14,adx=14,min_adx=12.0,max_adx=30.0,bb=20,bb_dev=2.0,max_bb_width_atr=5.0,
        min_atr=1.60,min_ema_slope_atr=0.025,pullback_atr=0.22,min_body_atr=0.12,
        sl_atr=1.00,tp_r=2.00,be_r=0.70,trail_start_r=1.20,trail_atr=0.60,
        risk_pct=0.50,max_lot=2.0,max_spread=1.00,max_spread_atr=0.45,
        start_hour=6,end_hour=20,friday_stop_hour=19,max_hold_sec=180,cooldown_sec=30)

class Strat(BaseStrat):
    def __init__(self,cfg):
        super().__init__(cfg);self.quality_pass=0;self.bos_pass=0;self.skipped_small_size=0;self.split_units=[]
    def on_bar(self,bar:Bar):
        x={'o':self.f(bar.open),'h':self.f(bar.high),'l':self.f(bar.low),'c':self.f(bar.close),'ts':int(bar.ts_event)};self.b.append(x)
        if self.entry is not None:return
        s=self.snap()
        if s is None:return
        self.setup_count+=1
        if s['atr']<P3['min_atr'] or not(P3['min_adx']<=s['adx']<=P3['max_adx']) or s['width']>P3['max_bb_width_atr']*s['atr']:return
        if abs(s['ema']-s['ema_prev'])<P3['min_ema_slope_atr']*s['atr']:return
        self.quality_pass+=1
        prev=list(self.b)[-2];body=abs(x['c']-x['o'])
        buy_dir=x['c']>s['ema'] and s['ema']>s['ema_prev'];sell_dir=x['c']<s['ema'] and s['ema']<s['ema_prev']
        if buy_dir or sell_dir:self.direction_count+=1
        buy_pull=x['l']<=s['ema']+P3['pullback_atr']*s['atr'] and x['c']>s['ema']
        sell_pull=x['h']>=s['ema']-P3['pullback_atr']*s['atr'] and x['c']<s['ema']
        if (buy_dir and buy_pull) or (sell_dir and sell_pull):self.pullback_count+=1
        buy_bos=x['c']>prev['h'];sell_bos=x['c']<prev['l']
        if (buy_dir and buy_pull and buy_bos) or (sell_dir and sell_pull and sell_bos):self.bos_pass+=1
        buy=buy_dir and buy_pull and buy_bos and x['c']>x['o'] and body>=P3['min_body_atr']*s['atr']
        sell=sell_dir and sell_pull and sell_bos and x['c']<x['o'] and body>=P3['min_body_atr']*s['atr']
        if buy:self.armed=('BUY',s)
        elif sell:self.armed=('SELL',s)
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);dt=pd.Timestamp(ts,unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed and self.entry is None and flat and ts>=self.cool_until:
            side,s=self.armed;spread=ask-bid
            if spread>P3['max_spread'] or spread>P3['max_spread_atr']*s['atr'] or not(P3['start_hour']<=dt.hour<P3['end_hour']) or (dt.weekday()==4 and dt.hour>=P3['friday_stop_hour']):return
            px=ask if side=='BUY' else bid;r=P3['sl_atr']*s['atr'];money=1000*P3['risk_pct']/100;theoretical=min(P3['max_lot']*100,money/r if r>0 else 0);legs=split_503020_integer(int(theoretical))
            if legs is None:self.skipped_small_size+=1;self.armed=None;return
            instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if side=='BUY' else OrderSide.SELL
            for units in legs:self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=instr.make_qty(Decimal(units))))
            self.split_units.append(legs);self.entry=px;self.risk=r;self.stop_ref=px-r if side=='BUY' else px+r;self.tp=px+P3['tp_r']*r if side=='BUY' else px-P3['tp_r']*r
            self.side=side;self.entry_ts=ts;self.exit_pending=False;self.baskets+=1;self.armed=None;return
        if self.entry is None or self.exit_pending:return
        px=bid if self.side=='BUY' else ask;profit=px-self.entry if self.side=='BUY' else self.entry-px;R=profit/self.risk if self.risk else 0;s=self.snap();atr=s['atr'] if s else self.risk/P3['sl_atr']
        if R>=P3['be_r']:
            be=self.entry+(0.02 if self.side=='BUY' else -0.02);self.stop_ref=max(self.stop_ref,be) if self.side=='BUY' else min(self.stop_ref,be)
        if R>=P3['trail_start_r']:
            ns=px-P3['trail_atr']*atr if self.side=='BUY' else px+P3['trail_atr']*atr;self.stop_ref=max(self.stop_ref,ns) if self.side=='BUY' else min(self.stop_ref,ns)
        hit=(px<=self.stop_ref or px>=self.tp) if self.side=='BUY' else (px>=self.stop_ref or px<=self.tp)
        timeout=(ts-self.entry_ts)>=P3['max_hold_sec']*1_000_000_000
        if hit or timeout:self.close_all_positions(self.config.instrument_id);self.exit_pending=True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks,replaced=ensure_executable_l1(raw,1000)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run();pos=eng.trader.generate_positions_report();tr=extract(pos);met=metrics(tr,days=int(man['days']));orders=eng.trader.generate_orders_report();fills=eng.trader.generate_order_fills_report();status={str(k):int(v) for k,v in orders['status'].astype(str).value_counts().to_dict().items()} if orders is not None and not orders.empty and 'status' in orders.columns else {}
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);summary=dict(verification_level='NAUTILUS_BT_RAW_BIDASK_L1_SIZE_ASSUMPTION',engine='NautilusTrader BacktestEngine',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),strategy='RangeHunter_M1_TrendFollow_v3',data_kind='RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes',ohlc_resample_used=False,period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),raw_tick_count=len(raw),l1_size_replacements=replaced,diagnostics=dict(bar_setups=st.setup_count,quality_pass=st.quality_pass,direction_pass=st.direction_count,pullback_pass=st.pullback_count,bos_pass=st.bos_pass,baskets_submitted=st.baskets,skipped_small_size=st.skipped_small_size,order_status=status,fill_rows=0 if fills is None else len(fills),position_rows=0 if pos is None else len(pos)),params=P3,metrics=met,limitations=['Raw Bid/Ask prices/timestamps unchanged; zero quote sizes replaced by synthetic L1 execution size 1000.','UTC session mapping.','No explicit commission/probabilistic slippage beyond native spread.'])
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False);orders.to_csv(out/'orders.csv',index=False) if orders is not None else None;fills.to_csv(out/'fills.csv',index=False) if fills is not None else None;(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'catalog_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
