#!/usr/bin/env python3
"""Jev-inspired directional decision on XAUUSD CFD bid/ask ticks. Paper replay only."""
import argparse
import csv
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def timestamp(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('tick time requires timezone')
    return parsed.astimezone(timezone.utc)


def run(path, capital=1000., lot=.01, contract_oz=100., lookback=20,
        threshold_usd=.5, max_spread_usd=.35, stop_usd=2., take_usd=4.,
        commission_round_usd=0., max_loss_pct=.05, synthetic=False,
        signal_mode='fixed_momentum', vol_k=1.5, min_efficiency=.45,
        spread_budget_fraction=.25, slippage_usd_per_side=0.,
        execution_delay_ticks=1):
    if not (capital>0 and lot>0 and contract_oz>0 and lookback>=2 and
            threshold_usd>0 and max_spread_usd>=0 and stop_usd>0 and take_usd>0 and
            commission_round_usd>=0 and 0<max_loss_pct<1 and vol_k>0 and
            0<=min_efficiency<=1 and 0<spread_budget_fraction<=1 and
            slippage_usd_per_side>=0 and execution_delay_ticks in (0,1) and
            signal_mode in ('fixed_momentum','adaptive_trend')):
        raise ValueError('invalid parameters')
    mids=deque(maxlen=lookback+1)
    cash=capital; peak=capital; maxdd=0.; min_equity=capital
    opened=None; pending=None; records=[]; prev=None; processed=0; blocked_spread=0
    halted=False; last_bid=last_ask=None; blocked_noise=0
    with open(path,newline='',encoding='utf-8') as f:
      reader=csv.DictReader(f)
      if not {'symbol','available_at','bid','ask'}.issubset(reader.fieldnames or []):
          raise ValueError('required columns: symbol,available_at,bid,ask')
      for r in reader:
        if r.get('symbol')!='XAUUSD':raise ValueError('XAUUSD only')
        at=timestamp(r['available_at'])
        bid=float(r['bid']);ask=float(r['ask'])
        if prev is not None and at<prev:raise ValueError('ticks must be non-decreasing')
        prev=at
        if not (math.isfinite(bid) and math.isfinite(ask) and bid>0 and ask>=bid):
            raise ValueError('invalid bid/ask')
        processed+=1
        last_bid,last_ask=bid,ask
        mid=(bid+ask)/2; mids.append(mid)
        exited=False
        entered=False
        if pending is not None:
            if ask-bid<=max_spread_usd:
                side=pending
                opened={'side':side,'entry':ask+slippage_usd_per_side if side=='BUY'
                        else bid-slippage_usd_per_side,'at':at}
                entered=True
            else:blocked_spread+=1
            pending=None
        if opened is not None:
            side=opened['side']; entry=opened['entry']
            close=bid-slippage_usd_per_side if side=='BUY' else ask+slippage_usd_per_side
            move=(close-entry) if side=='BUY' else (entry-close)
            floating=cash+move*lot*contract_oz-commission_round_usd
            peak=max(peak,floating);min_equity=min(min_equity,floating)
            maxdd=max(maxdd,1-floating/peak)
            risk_breach=floating<=capital*(1-max_loss_pct)
            if move<=-stop_usd or move>=take_usd or risk_breach:
                pnl=move*lot*contract_oz-commission_round_usd
                cash+=pnl
                records.append({'opened_at':opened['at'].isoformat(),'closed_at':at.isoformat(),
                                'side':side,'entry':entry,'exit':close,'net_pnl':pnl,
                                'reason':'RISK_LIMIT' if risk_breach else 'STOP' if move<=-stop_usd else 'TAKE'})
                opened=None
                exited=True
                if risk_breach:halted=True
        if cash<=capital*(1-max_loss_pct):halted=True
        if halted:break
        # A close on this tick cannot immediately be followed by a fresh entry.
        if opened is None and not exited and not entered and len(mids)==mids.maxlen:
            if ask-bid>max_spread_usd:
                blocked_spread+=1
            else:
                move=mids[-1]-mids[0]
                magnitude=abs(move)
                if signal_mode=='adaptive_trend':
                    path_length=sum(abs(mids[i]-mids[i-1]) for i in range(1,len(mids)))
                    efficiency=magnitude/path_length if path_length else 0.
                    volatility_floor=vol_k*path_length/math.sqrt(lookback)
                    eligible=(magnitude>=max(threshold_usd,volatility_floor) and
                              efficiency>=min_efficiency and
                              ask-bid<=spread_budget_fraction*magnitude)
                    if not eligible:blocked_noise+=1
                else:eligible=magnitude>=threshold_usd
                side=('BUY' if move>0 else 'SELL') if eligible else None
                if side:
                    if execution_delay_ticks:
                        pending=side
                    else:
                        opened={'side':side,'entry':ask+slippage_usd_per_side if side=='BUY'
                                else bid-slippage_usd_per_side,'at':at}
        peak=max(peak,cash); min_equity=min(min_equity,cash)
    if not processed:raise ValueError('empty tick file')
    if opened is not None:
        close=last_bid-slippage_usd_per_side if opened['side']=='BUY' else last_ask+slippage_usd_per_side
        move=(close-opened['entry']) if opened['side']=='BUY' else (opened['entry']-close)
        pnl=move*lot*contract_oz-commission_round_usd
        cash+=pnl
        min_equity=min(min_equity,cash)
        maxdd=max(maxdd,1-cash/peak)
        records.append({'opened_at':opened['at'].isoformat(),'closed_at':prev.isoformat(),
                        'side':opened['side'],'entry':opened['entry'],'exit':close,
                        'net_pnl':pnl,'reason':'END_OF_DATA'})
        opened=None
    gains=sum(max(0,r['net_pnl']) for r in records)
    losses=-sum(min(0,r['net_pnl']) for r in records)
    return {'label':'XAUUSD_TICK_RESEARCH_NOT_JEV_OR_HARLF_PARITY',
            'data_class':'SYNTHETIC_SMOKE_NO_MARKET_KPI' if synthetic else 'USER_SUPPLIED_TICKS_UNVERIFIED',
            'ticks':processed,
            'closed_trades':len(records),'gross_profit':gains,'gross_loss':losses,
            'profit_factor':gains/losses if losses else None,
            'win_rate':sum(r['net_pnl']>0 for r in records)/len(records) if records else None,
            'cash':cash,'last_marked_equity':cash,'min_marked_equity':min_equity,
            'max_marked_equity_dd':maxdd,'spread_blocked_ticks':blocked_spread,
            'noise_blocked_ticks':blocked_noise,
            'signal_mode':signal_mode,'signal_parameters':{
                'lookback':lookback,'threshold_usd':threshold_usd,'vol_k':vol_k,
                'min_efficiency':min_efficiency,'spread_budget_fraction':spread_budget_fraction},
            'open_position':False,'risk_halted':halted,'trade_log':records,
            'pending_at_end_canceled':pending is not None,
            'execution_delay_ticks':execution_delay_ticks,
            'slippage_usd_per_side':slippage_usd_per_side,
            'promotion':'BLOCKED_DATA_AND_INDEPENDENT_OOS',
            'caveat':'No L2 maker execution, no MON-USDC logic parity. Tick intervals between observations and source quality remain unverified.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ticks',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--capital',type=float,default=1000.)
    ap.add_argument('--lot',type=float,default=.01)
    ap.add_argument('--contract-oz',type=float,default=100.)
    ap.add_argument('--max-loss-pct',type=float,default=.05)
    ap.add_argument('--lookback',type=int,default=20)
    ap.add_argument('--threshold-usd',type=float,default=.5)
    ap.add_argument('--max-spread-usd',type=float,default=.35)
    ap.add_argument('--stop-usd',type=float,default=2.)
    ap.add_argument('--take-usd',type=float,default=4.)
    ap.add_argument('--commission-round-usd',type=float,default=0.)
    ap.add_argument('--signal-mode',choices=['fixed_momentum','adaptive_trend'],default='fixed_momentum')
    ap.add_argument('--vol-k',type=float,default=1.5)
    ap.add_argument('--min-efficiency',type=float,default=.45)
    ap.add_argument('--spread-budget-fraction',type=float,default=.25)
    ap.add_argument('--slippage-usd-per-side',type=float,default=0.)
    ap.add_argument('--execution-delay-ticks',type=int,choices=[0,1],default=1)
    ap.add_argument('--synthetic',action='store_true',help='Label input as synthetic; never count as market KPI')
    args=ap.parse_args()
    result=run(args.ticks,args.capital,lot=args.lot,contract_oz=args.contract_oz,
               max_loss_pct=args.max_loss_pct,lookback=args.lookback,threshold_usd=args.threshold_usd,
               max_spread_usd=args.max_spread_usd,stop_usd=args.stop_usd,take_usd=args.take_usd,
               commission_round_usd=args.commission_round_usd,synthetic=args.synthetic,
               signal_mode=args.signal_mode,vol_k=args.vol_k,
               min_efficiency=args.min_efficiency,spread_budget_fraction=args.spread_budget_fraction,
               slippage_usd_per_side=args.slippage_usd_per_side,
               execution_delay_ticks=args.execution_delay_ticks)
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='trade_log'},indent=2))
