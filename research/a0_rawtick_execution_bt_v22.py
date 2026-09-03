from __future__ import annotations
import json, os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from research.fib_ict_rawtick_detectors_v22 import Quote,RawTickConfig,detect_a0,raw_entry_price,raw_exit_price

@dataclass
class Trade:
 side:str; setup_ts:float; entry_ts:float; exit_ts:float; entry:float; exit:float; stop:float; target:float; pnl:float; r_multiple:float; reason:str; primary:float

def load_quotes(path:str)->list[Quote]:
 out=[]
 with open(path,encoding='utf-8') as f:
  for line in f:
   z=json.loads(line); out.append(Quote(datetime.fromisoformat(z['ts'].replace('Z','+00:00')).timestamp(),float(z['bid']),float(z['ask'])))
 return out

def first_after(qs,ts):
 lo=0; hi=len(qs)
 while lo<hi:
  m=(lo+hi)//2
  if qs[m].ts<=ts: lo=m+1
  else: hi=m
 return lo

def run(qs:list[Quote]):
 cfg=RawTickConfig(300,180,60,60)
 setups=detect_a0(qs,cfg); trades=[]
 for s in setups:
  i=first_after(qs,s.confirmed_at); zone_lo,zone_hi=s.ote_zone
  # Causal trigger: OTE area must be reached after Fib is armed, then executable quote must reclaim primary in thesis direction.
  touched=False; touch_i=None
  expiry=s.confirmed_at+1800
  while i<len(qs) and qs[i].ts<=expiry:
   q=qs[i]
   obs=q.ask if s.side=='LONG' else q.bid
   if zone_lo<=obs<=zone_hi:
    touched=True; touch_i=i; break
   # thesis invalid before entry
   if (s.side=='LONG' and q.bid<=s.sweep_extreme) or (s.side=='SHORT' and q.ask>=s.sweep_extreme): break
   i+=1
  if not touched: continue
  # Trigger is a post-arrival reclaim of the strategy primary, never the touch itself.
  j=touch_i+1; entry_i=None
  while j<len(qs) and qs[j].ts<=expiry:
   q=qs[j]
   if (s.side=='LONG' and q.bid<=s.sweep_extreme) or (s.side=='SHORT' and q.ask>=s.sweep_extreme): break
   if s.side=='LONG' and q.ask>=s.ote_primary: entry_i=j; break
   if s.side=='SHORT' and q.bid<=s.ote_primary: entry_i=j; break
   j+=1
  if entry_i is None: continue
  eq=qs[entry_i]; entry=raw_entry_price(s.side,eq); stop=s.sweep_extreme
  risk=(entry-stop) if s.side=='LONG' else (stop-entry)
  if risk<=0: continue
  target=entry+1.5*risk if s.side=='LONG' else entry-1.5*risk
  k=entry_i+1; exitq=None; reason='TIME'
  exit_deadline=eq.ts+3600
  while k<len(qs) and qs[k].ts<=exit_deadline:
   q=qs[k]
   px=raw_exit_price(s.side,q)
   if s.side=='LONG':
    if px<=stop: exitq=q; reason='SL'; break
    if px>=target: exitq=q; reason='TP'; break
   else:
    if px>=stop: exitq=q; reason='SL'; break
    if px<=target: exitq=q; reason='TP'; break
   k+=1
  if exitq is None: exitq=qs[min(k,len(qs)-1)]
  ex=raw_exit_price(s.side,exitq); pnl=(ex-entry) if s.side=='LONG' else (entry-ex)
  trades.append(Trade(s.side,s.confirmed_at,eq.ts,exitq.ts,entry,ex,stop,target,pnl,pnl/risk,reason,s.ote_primary))
 wins=[t for t in trades if t.pnl>0]; losses=[t for t in trades if t.pnl<0]
 gp=sum(t.pnl for t in wins); gl=-sum(t.pnl for t in losses)
 net=sum(t.pnl for t in trades); pf=gp/gl if gl else (float('inf') if gp else 0.0)
 eq=0.; peak=0.; maxdd=0.
 for t in trades:
  eq+=t.pnl; peak=max(peak,eq); maxdd=max(maxdd,peak-eq)
 return setups,trades,{'setups':len(setups),'trades':len(trades),'WR':100*len(wins)/len(trades) if trades else 0,'PF':pf,'EV_price':net/len(trades) if trades else 0,'Net_price':net,'MaxDD_price':maxdd,'avg_R':sum(t.r_multiple for t in trades)/len(trades) if trades else 0,'TP':sum(t.reason=='TP' for t in trades),'SL':sum(t.reason=='SL' for t in trades),'TIME':sum(t.reason=='TIME' for t in trades),'input':'RAW_BID_ASK_QUOTETICK_ONLY','ohlc':False,'entry':'POST_OTE_PRIMARY_RECLAIM','RR':1.5}

if __name__=='__main__':
 qs=load_quotes(os.environ.get('RAW_QUOTES','data/xauusd_quotes.jsonl')); setups,trades,kpi=run(qs)
 out=Path(os.environ.get('RESULT_DIR','results/a0-v22')); out.mkdir(parents=True,exist_ok=True)
 (out/'summary.json').write_text(json.dumps(kpi,indent=2),encoding='utf-8')
 (out/'trades.json').write_text(json.dumps([asdict(t) for t in trades],indent=2),encoding='utf-8')
 print(json.dumps(kpi,indent=2))
