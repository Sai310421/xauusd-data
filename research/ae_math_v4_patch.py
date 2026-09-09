from pathlib import Path
p=Path('research/ae_highn_2edge_math_raw6x3_bt.py')
s=p.read_text()
s=s.replace('import argparse,json,math','import argparse,json,math,os')
s=s.replace('from nautilus_trader.model.enums import AccountType,OmsType,OrderSide','from nautilus_trader.model.enums import AccountType,BookType,OmsType,OrderSide')
s=s.replace('from nautilus_trader.model.identifiers import InstrumentId','from nautilus_trader.model.identifiers import InstrumentId\nfrom nautilus_trader.model.objects import Quantity')
s=s.replace('cat.query_quote_ticks(identifiers=[inst.id.value])','cat.quote_ticks(instrument_ids=[inst.id.value])')
s=s.replace('e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))','e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))')
old='def run(inst,ticks,sym,tf,mode,days):\n cfg='
new='def run(inst,ticks,sym,tf,mode,days):\n zb=sum(1 for t in ticks if t.bid_size.as_double()<=0); za=sum(1 for t in ticks if t.ask_size.as_double()<=0)\n if zb or za:\n  one=Quantity.from_int(1)\n  ticks=[QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=t.bid_size if t.bid_size.as_double()>0 else one,ask_size=t.ask_size if t.ask_size.as_double()>0 else one,ts_event=t.ts_event,ts_init=t.ts_init) for t in ticks]\n cfg='
if old not in s: raise SystemExit('run target missing')
s=s.replace(old,new)
old='self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.'
new='self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.; self.trades=[]; self.fills=0; self.rejects=0; self.opened=0; self.closed=0; self.pending=False; self.reject_reasons={}; self.entry_px=None; self.debt_hist=deque(maxlen=128); self.ds_hist=[]; self.pnr_hist=[]; self.interventions=0; self.natural_waits=0; self.math_tail_exits=0; self.normal_exits=0; self.exit_mode=os.getenv("EXIT_MODE","A_EXIT").upper()'
if old not in s: raise SystemExit('init target missing')
s=s.replace(old,new)
s=s.replace('if self.portfolio.is_net_flat(self.config.instrument_id):','if self.cache.positions_open_count(instrument_id=self.config.instrument_id) == 0 and not self.pending:')
old='self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return'
new='self.pending=True; self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return'
s=s.replace(old,new)
old='bid,ask=self.f(t.bid_price),self.f(t.ask_price); A=self.h.on(Tick(float(t.ts_event),bid,ask,self.f(t.bid_size),self.f(t.ask_size))); B=self.B; act,val,ds,p,inter,adv=self.ctl.decide(A,B,self.debt,0,.8,-self.debt); self.actions[act.value]=self.actions.get(act.value,0)+1'
new='bid,ask=self.f(t.bid_price),self.f(t.ask_price); spread=ask-bid; A=self.h.on(Tick(float(t.ts_event),bid,ask,self.f(t.bid_size),self.f(t.ask_size))); B=self.B; floatp=0. if self.entry_px is None or self.direction==0 else ((bid-self.entry_px) if self.direction>0 else (self.entry_px-ask))*float(self.config.trade_size); self.debt=max(0.,-floatp); self.debt_hist.append(self.debt); dh=np.diff(np.array(self.debt_hist,float)) if len(self.debt_hist)>2 else np.array([0.]); mu=float(dh.mean()) if len(dh) else 0.; sig=max(.05,float(dh.std())*8 if len(dh)>2 else .8); act,val,ds0,p0,inter0,adv=self.ctl.decide(A,B,self.debt,mu,sig,floatp); ds=max(ds0,2.0*spread); tailB=max(1.0,8.0*ds); p=pnr(self.debt,mu,sig,tailB); inter=max(0.,self.debt-ds); self.ds_hist.append(ds); self.pnr_hist.append(p); self.actions[act.value]=self.actions.get(act.value,0)+1; self.natural_waits += int(self.debt<ds or p>=.35)'
if old not in s: raise SystemExit('quote target missing')
s=s.replace(old,new)
old='else:d=1 if act==Action.ENTRY_LONG else -1 if act==Action.ENTRY_SHORT else 0'
new='else:\n   ea=A.score*A.expected_move-A.cost if A.direction else -1e9; eb=B.score*B.expected_move-B.cost if B.direction else -1e9; best=A if ea>=eb else B; d=best.direction if max(ea,eb)>0 else 0'
if old not in s: raise SystemExit('direction target missing')
s=s.replace(old,new)
old='if mode=="AB100D" and act in (Action.EXIT,Action.REDUCE,Action.HEDGE): self.close_all_positions(self.config.instrument_id)'
new='if mode=="AB100D":\n   tail_intervene=(self.debt>=ds and p<.35 and inter>0)\n   ea=A.score*A.expected_move-A.cost if A.direction else -1e9; eb=B.score*B.expected_move-B.cost if B.direction else -1e9; best=A if ea>=eb else B\n   a_rev=((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0))\n   b_rev=((self.direction>0 and B.direction<0) or (self.direction<0 and B.direction>0))\n   ab_rev=(best.direction!=0 and best.direction==-self.direction and max(ea,eb)>0)\n   normal_reverse=a_rev if self.exit_mode=="A_EXIT" else b_rev if self.exit_mode=="B_EXIT" else ab_rev if self.exit_mode=="AB_EXIT" else False\n   if tail_intervene:\n    self.interventions+=1; self.math_tail_exits+=1; self.close_all_positions(self.config.instrument_id)\n   elif normal_reverse:\n    self.normal_exits+=1; self.close_all_positions(self.config.instrument_id)'
if old not in s: raise SystemExit('exit target missing')
s=s.replace(old,new)
old=' def on_position_closed(self,e): self.direction=0'
new=' def on_order_filled(self,e):\n  self.fills+=1; self.pending=False\n  if self.entry_px is None and self.direction!=0: self.entry_px=self.f(e.last_px)\n def _reject(self,e):\n  self.rejects+=1; self.pending=False; r=str(getattr(e,"reason","UNKNOWN")); self.reject_reasons[r]=self.reject_reasons.get(r,0)+1\n def on_order_rejected(self,e): self._reject(e)\n def on_order_denied(self,e): self._reject(e)\n def on_order_canceled(self,e): self.pending=False\n def on_position_opened(self,e):\n  self.opened+=1; self.pending=False; self.entry_px=self.f(e.avg_px_open)\n def on_position_closed(self,e):\n  self.direction=0; self.closed+=1; self.pending=False; self.entry_px=None; self.debt=0.; self.debt_hist.clear(); self.trades.append({"pnl":parse_money(e.realized_pnl),"ts_closed":int(e.ts_closed or e.ts_event)})'
if old not in s: raise SystemExit('event target missing')
s=s.replace(old,new)
old='e.run(); trades=extract(e.generate_positions_report(),sym,tf); m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
new='e.run(); trades=[{"symbol":sym,"tf":tf,**x} for x in s.trades]; m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"fills":s.fills,"rejected":s.rejects,"opened":s.opened,"closed":s.closed,"reject_reasons":s.reject_reasons,"exit_mode":s.exit_mode,"interventions":s.interventions,"math_tail_exits":s.math_tail_exits,"normal_exits":s.normal_exits,"natural_waits":s.natural_waits,"Dstar_avg":float(np.mean(s.ds_hist)) if s.ds_hist else 0.,"pNR_avg":float(np.mean(s.pnr_hist)) if s.pnr_hist else 0.,"zero_bid_sizes":zb,"zero_ask_sizes":za,"liquidity_size_floor":1,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
if old not in s: raise SystemExit('tail target missing')
s=s.replace(old,new)
s=s.replace('for mode in ["A","B","AB","AB100D"]:', 'for mode in ["AB100D"]:')
p.write_text(s)
compile(s,str(p),'exec')
print('AE_MATH_V4_PATCH_OK')
