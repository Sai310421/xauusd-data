#!/usr/bin/env python3
"""
TEST-BOT-TRADING-MT5 native-logic bridge.

Preserves the original TEST-BOT backtest_m1_v3.py strategy/position logic and constants:
LOT=0.01, SL_M=0.3, TP_M=0.6, TRAIL_M=0.2,
EMA8/EMA34, ATR14, RSI7, momentum(5), single active trade,
5 consecutive-loss stop, and the original three strategy modes.

Only the MT5 data adapter is replaced. Data comes from the repository's raw
Dukascopy Bid/Ask QuoteTick loader and M1 bars are aggregated directly from
raw ticks (not from OHLC resampling). This lets the original BOT logic run on
GitHub Actions/Linux without MetaTrader5 installed.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import bigplayer_synergy_supervisor_21d as core

LOT = 0.01
SL_M = 0.3
TP_M = 0.6
TRAIL_M = 0.2
INITIAL_BALANCE = 1000.0


def raw_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    t = ticks.copy()
    t['mid'] = (t['bid'] + t['ask']) / 2.0
    x = t.set_index('datetime')['mid'].resample('1min').ohlc().dropna()
    # spread is diagnostic only; native v3 logic does not filter on spread.
    spr = (t.set_index('datetime')['ask'] - t.set_index('datetime')['bid']).resample('1min').mean()
    x['spread'] = spr.reindex(x.index).fillna(0.0)
    return x


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df['close'], df['high'], df['low']
    df['ema8'] = c.ewm(span=8, adjust=False).mean()
    df['ema34'] = c.ewm(span=34, adjust=False).mean()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=7, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=7, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['mom'] = c.pct_change(5)
    return df.dropna()


def native_run(df: pd.DataFrame, strat: str):
    trades=[]; active=None; cons_loss=0
    equity=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    for idx in range(1, len(df)):
        bar=df.iloc[idx]; prev=df.iloc[idx-1]; ts=df.index[idx]
        if cons_loss >= 5:
            continue
        tu=bar['ema8'] > bar['ema34']; td=bar['ema8'] < bar['ema34']
        if active:
            if TRAIL_M > 0:
                trail_dist=active['atr'] * TRAIL_M
                if active['side']=='BUY':
                    pf=bar['close']-active['entry']
                    if pf > trail_dist and bar['close']-trail_dist > active['sl']:
                        active['sl']=bar['close']-trail_dist
                else:
                    pf=active['entry']-bar['close']
                    if pf > trail_dist and bar['close']+trail_dist < active['sl']:
                        active['sl']=bar['close']+trail_dist
            ep=er=None
            if active['side']=='BUY':
                if bar['high'] >= active['tp']: ep,er=active['tp'],'TP'
                elif bar['low'] <= active['sl']: ep,er=active['sl'],'SL'
            else:
                if bar['low'] <= active['tp']: ep,er=active['tp'],'TP'
                elif bar['high'] >= active['sl']: ep,er=active['sl'],'SL'
            if ep is not None:
                pnl=(ep-active['entry'])*LOT*100 if active['side']=='BUY' else (active['entry']-ep)*LOT*100
                trades.append({'entry_time':active['time'],'exit_time':ts,'side':active['side'],'entry':active['entry'],'exit':ep,'pnl':pnl,'reason':er})
                equity += pnl; peak=max(peak,equity); max_dd=max(max_dd, peak-equity)
                if pnl > 0: cons_loss=0
                else: cons_loss += 1
                active=None
                continue
            else:
                continue
        sig=None
        if strat=='EMA_CROSS':
            if prev['ema8'] <= prev['ema34'] and bar['ema8'] > bar['ema34']: sig='BUY'
            elif prev['ema8'] >= prev['ema34'] and bar['ema8'] < bar['ema34']: sig='SELL'
        elif strat=='RSI_TREND':
            if tu and bar['rsi'] < 30: sig='BUY'
            elif td and bar['rsi'] > 70: sig='SELL'
        elif strat=='MOMENTUM':
            if bar['mom'] > 0.0005 and tu: sig='BUY'
            elif bar['mom'] < -0.0005 and td: sig='SELL'
        if not sig: continue
        sl=round(bar['close']-bar['atr']*SL_M,2) if sig=='BUY' else round(bar['close']+bar['atr']*SL_M,2)
        tp=round(bar['close']+bar['atr']*TP_M,2) if sig=='BUY' else round(bar['close']-bar['atr']*TP_M,2)
        active={'side':sig,'entry':bar['close'],'sl':sl,'tp':tp,'atr':bar['atr'],'time':ts}
    if active:
        ep=float(df.iloc[-1]['close']); ts=df.index[-1]
        pnl=(ep-active['entry'])*LOT*100 if active['side']=='BUY' else (active['entry']-ep)*LOT*100
        trades.append({'entry_time':active['time'],'exit_time':ts,'side':active['side'],'entry':active['entry'],'exit':ep,'pnl':pnl,'reason':'END'})
        equity += pnl; peak=max(peak,equity); max_dd=max(max_dd,peak-equity)
    return pd.DataFrame(trades), max_dd


def metrics(trades: pd.DataFrame, max_dd: float, days: int):
    if len(trades)==0:
        return dict(N=0,N_per_day=0,WR=0,PF=0,RF=0,net_profit=0,return_pct=0,daily_return_pct=0,max_dd_pct=0,max_dd_usd=0,final_balance=INITIAL_BALANCE)
    p=trades.pnl.astype(float); gp=p[p>0].sum(); gl=-p[p<0].sum(); net=p.sum(); n=len(p)
    pf=gp/gl if gl>0 else math.inf
    wr=(p>0).mean()*100
    dd_pct=max_dd/INITIAL_BALANCE*100
    rf=net/max_dd if max_dd>0 else (math.inf if net>0 else 0)
    return dict(N=n,N_per_day=n/days,WR=wr,PF=pf,RF=rf,net_profit=net,return_pct=net/INITIAL_BALANCE*100,daily_return_pct=net/INITIAL_BALANCE*100/days,max_dd_pct=dd_pct,max_dd_usd=max_dd,final_balance=INITIAL_BALANCE+net)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--days',type=int,default=21); a=ap.parse_args()
    bdays=core.business_days(a.start,a.days)
    ticks=core.load_ticks(bdays)
    df=prepare(raw_m1(ticks))
    rows=[]; out=ROOT/'results'/'testbot_native_bridge_21d'; out.mkdir(parents=True,exist_ok=True)
    for strat in ['EMA_CROSS','RSI_TREND','MOMENTUM']:
        tr,dd=native_run(df,strat); tr.to_csv(out/f'trades_{strat}.csv',index=False)
        m=metrics(tr,dd,a.days); m['config']=f'TESTBOT_NATIVE_{strat}'; rows.append(m)
    s=pd.DataFrame(rows).sort_values(['return_pct','PF'],ascending=False)
    s.to_csv(out/'summary_21d.csv',index=False)
    (out/'provenance.json').write_text(json.dumps({'source':'TEST-BOT-TRADING-MT5-master(2).zip/backtest_m1_v3.py','logic':'preserved','data_adapter':'MT5 copy_rates_from_pos replaced by raw Dukascopy QuoteTick -> direct M1 aggregation','ohlc_resample_used':False,'initial_balance':INITIAL_BALANCE,'lot':LOT,'SL_M':SL_M,'TP_M':TP_M,'TRAIL_M':TRAIL_M,'days':a.days,'start':a.start},indent=2),encoding='utf-8')
    print(s.to_string(index=False))

if __name__=='__main__': main()
