from __future__ import annotations
import argparse,json
from collections import deque
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

P4=dict(ema=21,atr=14,m5_ema=21,m5_slope_atr=0.035,m5_structure_lookback=3,
        sweep_lookback=5,sweep_atr=0.06,bos_lookback=3,min_body_atr=0.08,
        tick_window_sec=8,tick_min_move_atr=0.025,tick_min_direction_ratio=0.58,
        sl_atr=0.85,tp_r=2.20,be_r=0.75,trail_start_r=1.30,trail_atr=0.65,
        risk_pct=0.50,max_lot=2.0,max_spread=1.0,max_spread_atr=0.40,
        start_hour=6,end_hour=22,friday_stop_hour=19,max_hold_sec=180,cooldown_sec=30)

class Strat(BaseStrat):
    def __init__(self,cfg):
        super().__init__(cfg);self.m1=deque(maxlen=240);self.m5=deque(maxlen=120);self.tickbuf=deque();self.m5_bias=None
        self.bias_pass=self.sweep_pass=self.bos_pass=self.tick_pass=self.skipped_small_size=0
    def _ema_list(self,vals,n): return self.ema(np.asarray(vals,float),n)
    def _atr_list(self,bars,n=14):
        if len(bars)<n+1:return None
        x=list(bars);tr=[max(x[i]['h']-x[i]['l'],abs(x[i]['h']-x[i-1]['c']),abs(x[i]['l']-x[i-1]['c'])) for i in range(1,len(x))]
        a=sum(tr[:n])/n
        for z in tr[n:]:a=(a*(n-1)+z)/n
        return float(a)
    def _update_m5(self,x):
        bucket=x['ts']//300_000_000_000
        if not self.m5 or self.m5[-1]['bucket']!=bucket:self.m5.append(dict(o=x['o'],h=x['h'],l=x['l'],c=x['c'],ts=x['ts'],bucket=bucket))
        else:
            q=self.m5[-1];q['h']=max(q['h'],x['h']);q['l']=min(q['l'],x['l']);q['c']=x['c'];q['ts']=x['ts']
    def _bias(self):
        if len(self.m5)<30:return None
        bars=list(self.m5)[:-1]
        if len(bars)<25:return None
        c=[z['c'] for z in bars];ema=self._ema_list(c,P4['m5_ema']);prev=self._ema_list(c[:-1],P4['m5_ema']);atr=self._atr_list(bars,P4['atr'])
        if None in (ema,prev,atr) or atr<=0:return None
        slope=(ema-prev)/atr;lb=P4['m5_structure_lookback'];last=bars[-1];prior=bars[-1-lb:-1]
        up_structure=last['h']>max(z['h'] for z in prior) or (last['h']>prior[-1]['h'] and last['l']>=prior[-1]['l'])
        dn_structure=last['l']<min(z['l'] for z in prior) or (last['l']<prior[-1]['l'] and last['h']<=prior[-1]['h'])
        if last['c']>ema and slope>=P4['m5_slope_atr'] and up_structure:return 'BUY',atr
        if last['c']<ema and slope<=-P4['m5_slope_atr'] and dn_structure:return 'SELL',atr
        return None
    def on_bar(self,bar:Bar):
        x={'o':self.f(bar.open),'h':self.f(bar.high),'l':self.f(bar.low),'c':self.f(bar.close),'ts':int(bar.ts_event)};self.b.append(x);self.m1.append(x);self._update_m5(x)
        if self.entry is not None:return
        self.setup_count+=1;bias=self._bias();self.m5_bias=bias
        if bias is None or len(self.m1)<10:return
        side,m5atr=bias;self.bias_pass+=1
        a=self.atr(P4['atr']);
        if a is None or a<=0:return
        hist=list(self.m1);prev=hist[-2];look=hist[-1-P4['sweep_lookback']:-1];bos_hist=hist[-1-P4['bos_lookback']:-1]
        if side=='BUY':
            swept=x['l']<min(z['l'] for z in look)-P4['sweep_atr']*a and x['c']>min(z['l'] for z in look)
            bos=x['c']>max(z['h'] for z in bos_hist) and x['c']>x['o'] and abs(x['c']-x['o'])>=P4['min_body_atr']*a
        else:
            swept=x['h']>max(z['h'] for z in look)+P4['sweep_atr']*a and x['c']<max(z['h'] for z in look)
            bos=x['c']<min(z['l'] for z in bos_hist) and x['c']<x['o'] and abs(x['c']-x['o'])>=P4['min_body_atr']*a
        # Permit either same-bar sweep/reclaim+BOS or a sweep on the immediately previous bar followed by BOS.
        if not swept:
            old=hist[-2-P4['sweep_lookback']:-2]
            swept=(prev['l']<min(z['l'] for z in old)-P4['sweep_atr']*a and prev['c']>min(z['l'] for z in old)) if side=='BUY' else (prev['h']>max(z['h'] for z in old)+P4['sweep_atr']*a and prev['c']<max(z['h'] for z in old))
        if swept:self.sweep_pass+=1
        if swept and bos:self.bos_pass+=1;self.armed=(side,dict(atr=a,m5atr=m5atr,signal_ts=x['ts']))
    def _tick_accel(self,side,atr,ts,bid,ask):
        mid=(bid+ask)/2;self.tickbuf.append((ts,mid));cut=ts-P4['tick_window_sec']*1_000_000_000
        while self.tickbuf and self.tickbuf[0][0]<cut:self.tickbuf.popleft()
        if len(self.tickbuf)<4:return False
        mids=[z[1] for z in self.tickbuf];diff=np.diff(mids);move=mids[-1]-mids[0];nz=diff[diff!=0]
        if len(nz)==0:return False
        ratio=float((nz>0).mean()) if side=='BUY' else float((nz<0).mean())
        ok=(move>=P4['tick_min_move_atr']*atr if side=='BUY' else move<=-P4['tick_min_move_atr']*atr) and ratio>=P4['tick_min_direction_ratio']
        return ok
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);dt=pd.Timestamp(ts,unit='ns',tz='UTC')
        flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed and self.entry is None and flat and ts>=self.cool_until:
            side,s=self.armed;spread=ask-bid
            if ts-s['signal_ts']>60_000_000_000:self.armed=None;return
            if spread>P4['max_spread'] or spread>P4['max_spread_atr']*s['atr'] or not(P4['start_hour']<=dt.hour<P4['end_hour']) or (dt.weekday()==4 and dt.hour>=P4['friday_stop_hour']):return
            if not self._tick_accel(side,s['atr'],ts,bid,ask):return
            self.tick_pass+=1;px=ask if side=='BUY' else bid;r=P4['sl_atr']*s['atr'];money=1000*P4['risk_pct']/100;theoretical=min(P4['max_lot']*100,money/r if r>0 else 0);legs=split_503020_integer(int(theoretical))
            if legs is None:self.skipped_small_size+=1;self.armed=None;return
            instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if side=='BUY' else OrderSide.SELL
            for units in legs:self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=instr.make_qty(Decimal(units))))
            self.entry=px;self.risk=r;self.stop_ref=px-r if side=='BUY' else px+r;self.tp=px+P4['tp_r']*r if side=='BUY' else px-P4['tp_r']*r;self.side=side;self.entry_ts=ts;self.exit_pending=False;self.baskets+=1;self.armed=None;return
        if self.entry is None or self.exit_pending:return
        px=bid if self.side=='BUY' else ask;profit=px-self.entry if self.side=='BUY' else self.entry-px;R=profit/self.risk if self.risk else 0;a=self.atr(P4['atr']) or self.risk/P4['sl_atr']
        if R>=P4['be_r']:
            be=self.entry+(0.02 if self.side=='BUY' else -0.02);self.stop_ref=max(self.stop_ref,be) if self.side=='BUY' else min(self.stop_ref,be)
        if R>=P4['trail_start_r']:
            ns=px-P4['trail_atr']*a if self.side=='BUY' else px+P4['trail_atr']*a;self.stop_ref=max(self.stop_ref,ns) if self.side=='BUY' else min(self.stop_ref,ns)
        hit=(px<=self.stop_ref or px>=self.tp) if self.side=='BUY' else (px>=self.stop_ref or px<=self.tp);timeout=(ts-self.entry_ts)>=P4['max_hold_sec']*1_000_000_000
        if hit or timeout:self.close_all_positions(self.config.instrument_id);self.exit_pending=True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks,replaced=ensure_executable_l1(raw,1000)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run();pos=eng.trader.generate_positions_report();tr=extract(pos);met=metrics(tr,days=int(man['days']));orders=eng.trader.generate_orders_report();fills=eng.trader.generate_order_fills_report();status={str(k):int(v) for k,v in orders['status'].astype(str).value_counts().to_dict().items()} if orders is not None and not orders.empty and 'status' in orders.columns else {}
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);summary=dict(verification_level='NAUTILUS_BT_RAW_BIDASK_L1_SIZE_ASSUMPTION',engine='NautilusTrader BacktestEngine',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),strategy='RangeHunter_M1_TrendFollow_v4_AdaptiveMTF',data_kind='RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes',ohlc_resample_used=False,logic='M5 bias -> M1 liquidity sweep/reclaim -> M1 micro BOS -> raw tick acceleration -> 50/30/20',period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),raw_tick_count=len(raw),l1_size_replacements=replaced,diagnostics=dict(bar_setups=st.setup_count,m5_bias_pass=st.bias_pass,sweep_pass=st.sweep_pass,bos_pass=st.bos_pass,tick_accel_pass=st.tick_pass,baskets_submitted=st.baskets,skipped_small_size=st.skipped_small_size,order_status=status,fill_rows=0 if fills is None else len(fills),position_rows=0 if pos is None else len(pos)),params=P4,metrics=met,limitations=['Raw Bid/Ask prices/timestamps unchanged; zero quote sizes replaced by synthetic L1 execution size 1000.','M5 is built directly from sequential M1 INTERNAL bars generated by Nautilus from the same raw ticks; no external OHLC/resample dataset fallback.','UTC session mapping.','No explicit commission/probabilistic slippage beyond native spread.'])
    pd.DataFrame(tr).to_csv(out/'trades.csv',index=False);orders.to_csv(out/'orders.csv',index=False) if orders is not None else None;fills.to_csv(out/'fills.csv',index=False) if fills is not None else None;(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'catalog_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
