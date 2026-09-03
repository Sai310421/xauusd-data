from pathlib import Path

p=Path('research/ae_harmonic_sequence_runner.py')
s=p.read_text()

# Raw L1 execution compatibility only.
s=s.replace('import argparse,json,math','import argparse,json,math')
s=s.replace('from nautilus_trader.model import BarType,Money,Venue','from nautilus_trader.model import BarType,Money,Venue\nfrom nautilus_trader.model.enums import BookType')
s=s.replace('from nautilus_trader.model.identifiers import InstrumentId','from nautilus_trader.model.identifiers import InstrumentId\nfrom nautilus_trader.model.objects import Quantity')
s=s.replace('cat.query_quote_ticks(identifiers=[inst.id.value])','cat.quote_ticks(instrument_ids=[inst.id.value])')
s=s.replace('e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))','e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))')

# Replace harmonic scorer with ordered X-A-B-C-D validation using standard AD/XA geometry.
start=s.index('class Harm:')
end=s.index('def ystar', start)
new_harm='''class Harm:\n def on(self,x,a,b,c,d,d_kind,cost=0.,pattern_id=None):\n  xa,ab,bc,cd=abs(a-x),abs(b-a),abs(c-b),abs(d-c)\n  if min(xa,ab,bc,cd)<=1e-12:return Sig("B",0,0,0,cost)\n  rb=ab/xa; rc=bc/ab; adxa=abs(d-a)/xa; rcd=cd/bc\n  checks=[]\n  # Gartley: AB~0.618, BC 0.382-0.886, AD~0.786, CD 1.13-1.618\n  g=(abs(rb-.618)<=.08 and .382<=rc<=.886 and abs(adxa-.786)<=.10 and 1.13<=rcd<=1.70)\n  # Butterfly: AB~0.786, BC 0.382-0.886, AD 1.27-1.618, CD 1.618-2.618\n  bf=(abs(rb-.786)<=.08 and .382<=rc<=.886 and 1.20<=adxa<=1.70 and 1.50<=rcd<=2.75)\n  # Bat: AB 0.382-0.50, BC 0.382-0.886, AD~0.886, CD 1.618-2.618\n  bat=(.36<=rb<=.52 and .382<=rc<=.886 and abs(adxa-.886)<=.10 and 1.50<=rcd<=2.75)\n  if g: name='G'; score=1.0\n  elif bf: name='BF'; score=1.0\n  elif bat: name='BAT'; score=1.0\n  else:return Sig("B",0,0,0,cost,{"rb":rb,"rc":rc,"adxa":adxa,"rcd":rcd,"pattern_id":pattern_id})\n  dr=1 if d_kind=='L' else -1 if d_kind=='H' else 0\n  return Sig("B"+name,dr,score,abs(c-d),cost,{"rb":rb,"rc":rc,"adxa":adxa,"rcd":rcd,"pattern_id":pattern_id})\n\nclass SwingSeq:\n def __init__(self,depth=2):\n  self.depth=depth; self.bars=[]; self.pivots=[]; self.seq_id=0\n def on_bar(self,b):\n  self.bars.append(b)\n  n=2*self.depth+1\n  if len(self.bars)<n:return None\n  w=self.bars[-n:]; c=w[self.depth]\n  highs=[z['h'] for z in w]; lows=[z['l'] for z in w]\n  kind=None; price=None\n  if c['h']==max(highs) and highs.count(c['h'])==1: kind='H'; price=c['h']\n  if c['l']==min(lows) and lows.count(c['l'])==1:\n   if kind is not None:return None\n   kind='L'; price=c['l']\n  if kind is None:return None\n  pivot={'kind':kind,'price':price,'bar_index':len(self.bars)-self.depth-1}\n  if self.pivots and self.pivots[-1]['kind']==kind:\n   old=self.pivots[-1]\n   more=(kind=='H' and price>old['price']) or (kind=='L' and price<old['price'])\n   if more:self.pivots[-1]=pivot\n   return None\n  self.pivots.append(pivot); self.pivots=self.pivots[-12:]\n  if len(self.pivots)<5:return None\n  q=self.pivots[-5:]\n  kinds=''.join(z['kind'] for z in q)\n  if kinds not in ('HLHLH','LHLHL'):return None\n  self.seq_id+=1\n  return q,self.seq_id\n\n'''
s=s[:start]+new_harm+s[end:]

# Replace strategy B construction: confirmed alternating swing pivots only, one signal per completed sequence.
s=s.replace('self.h=HFT(); self.hm=Harm(); self.ctl=Controller(); self.bs=deque(maxlen=32); self.B=Sig("B",0,0,0,0); self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.', 'self.h=HFT(); self.hm=Harm(); self.swing=SwingSeq(depth=2); self.bs=deque(maxlen=32); self.B=Sig("B",0,0,0,0); self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.; self.last_b_entry_id=None; self.trades=[]; self.fills=0; self.rejects=0; self.opened=0; self.closed=0; self.pending=False; self.reject_reasons={}')
old_bar=''' def on_bar(self,b):\n  x={"o":self.f(b.open),"h":self.f(b.high),"l":self.f(b.low),"c":self.f(b.close)}; self.bs.append(x)\n  if len(self.bs)>=10:\n   z=list(self.bs); p=[z[-10]["c"],z[-8]["c"],z[-6]["c"],z[-4]["c"],z[-2]["c"]]; bias=float(np.sign(z[-2]["c"]-z[-8]["c"])); self.B=self.hm.on(*p,bias=bias,cost=0.)\n'''
new_bar=''' def on_bar(self,b):\n  x={"o":self.f(b.open),"h":self.f(b.high),"l":self.f(b.low),"c":self.f(b.close)}; self.bs.append(x)\n  got=self.swing.on_bar(x)\n  if got is not None:\n   q,pid=got; vals=[z["price"] for z in q]; self.B=self.hm.on(*vals,d_kind=q[-1]["kind"],cost=0.,pattern_id=pid)\n  else:\n   # Keep the last valid completed pattern until consumed; do not fabricate intermediate XABCD points.\n   pass\n'''
if old_bar not in s: raise SystemExit('on_bar target missing')
s=s.replace(old_bar,new_bar)

# Stable fill accounting + one-entry-per-completed-harmonic-pattern.
s=s.replace('if self.portfolio.is_net_flat(self.config.instrument_id):', 'if self.cache.positions_open_count(instrument_id=self.config.instrument_id) == 0 and not self.pending:')
s=s.replace('''  if mode=="A": d=A.direction\n  elif mode=="B": d=B.direction\n  elif mode=="AB": d=(A if A.score*A.expected_move-A.cost>=B.score*B.expected_move-B.cost else B).direction\n''','''  if mode=="A": d=A.direction\n  elif mode=="B":\n   pid=B.meta.get("pattern_id") if B.meta else None; d=B.direction if pid is not None and pid!=self.last_b_entry_id else 0\n  elif mode=="AB": d=(A if A.score*A.expected_move-A.cost>=B.score*B.expected_move-B.cost else B).direction\n''')
s=s.replace('self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return', 'self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; self.pending=True;\n   if mode=="B" and B.meta: self.last_b_entry_id=B.meta.get("pattern_id")\n   return')

ev_old=' def on_position_closed(self,e): self.direction=0'
ev_new=''' def on_order_filled(self,e): self.fills+=1; self.pending=False\n def _reject(self,e):\n  self.rejects+=1; self.pending=False\n  r=str(getattr(e,"reason","UNKNOWN")); self.reject_reasons[r]=self.reject_reasons.get(r,0)+1\n def on_order_rejected(self,e): self._reject(e)\n def on_order_denied(self,e): self._reject(e)\n def on_order_canceled(self,e): self.pending=False\n def on_position_opened(self,e): self.opened+=1; self.pending=False\n def on_position_closed(self,e):\n  self.direction=0; self.closed+=1; self.pending=False\n  self.trades.append({"pnl":parse_money(e.realized_pnl),"ts_closed":int(e.ts_closed or e.ts_event)})'''
if ev_old not in s: raise SystemExit('events target missing')
s=s.replace(ev_old,ev_new)

run_old='def run(inst,ticks,sym,tf,mode,days):\n cfg='
run_new='def run(inst,ticks,sym,tf,mode,days):\n zb=sum(1 for t in ticks if t.bid_size.as_double()<=0); za=sum(1 for t in ticks if t.ask_size.as_double()<=0)\n if zb or za:\n  one=Quantity.from_int(1)\n  ticks=[QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=t.bid_size if t.bid_size.as_double()>0 else one,ask_size=t.ask_size if t.ask_size.as_double()>0 else one,ts_event=t.ts_event,ts_init=t.ts_init) for t in ticks]\n cfg='
if run_old not in s: raise SystemExit('run target missing')
s=s.replace(run_old,run_new)
old='e.run(); trades=extract(e.generate_positions_report(),sym,tf); m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
new='e.run(); trades=[{"symbol":sym,"tf":tf,**x} for x in s.trades]; m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"fills":s.fills,"rejected":s.rejects,"opened":s.opened,"closed":s.closed,"reject_reasons":s.reject_reasons,"zero_bid_sizes":zb,"zero_ask_sizes":za,"liquidity_size_floor":1,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
if old not in s: raise SystemExit('metrics target missing')
s=s.replace(old,new)

# One mode/engine per process; this validation lane is B-only.
needle='ap.add_argument("--experiment-id",required=True); ap.add_argument("--raw-bidask-only",action="store_true"); a=ap.parse_args()'
repl='ap.add_argument("--experiment-id",required=True); ap.add_argument("--mode",choices=["B"],default="B"); ap.add_argument("--raw-bidask-only",action="store_true"); a=ap.parse_args()'
s=s.replace(needle,repl)
s=s.replace('all_by_mode={m:[] for m in ["A","B","AB","AB100D"]}', 'all_by_mode={"B":[]}')
s=s.replace('for mode in ["A","B","AB","AB100D"]:', 'for mode in ["B"]:')

p.write_text(s)
compile(s,str(p),'exec')
print('AE_HARMONIC_SEQUENCE_PATCH_OK')
