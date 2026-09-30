from pathlib import Path
p=Path('research/ae_highn_2edge_math_raw6x3_bt.py')
s=p.read_text()
old='self.exit_mode=os.getenv("EXIT_MODE","A_EXIT").upper()'
new='self.exit_mode=os.getenv("EXIT_MODE","B_EXIT").upper(); self.entry_variant=os.getenv("ENTRY_VARIANT","OWNED_EXIT").upper(); self.pending_edge=None; self.active_edge=None; self.edge_entries={"A":0,"B":0}; self.entry_gate_blocked=0'
if old not in s: raise SystemExit('v5 init target missing')
s=s.replace(old,new)
old='else:\n   ea=A.score*A.expected_move-A.cost if A.direction else -1e9; eb=B.score*B.expected_move-B.cost if B.direction else -1e9; best=A if ea>=eb else B; d=best.direction if max(ea,eb)>0 else 0'
new='else:\n   ae=A.score*A.expected_move-spread if A.direction else -1e9; be=B.score*B.expected_move-spread if B.direction else -1e9\n   apass=(A.direction!=0 and A.expected_move>1.5*spread and ae>0)\n   bpass=(B.direction!=0 and B.score>=.60 and B.expected_move>1.5*spread and be>0)\n   selected_edge=None; d=0\n   if self.entry_variant=="B_ONLY":\n    if bpass: selected_edge="B"; d=B.direction\n   elif self.entry_variant=="A_TO_B":\n    if apass: selected_edge="A"; d=A.direction\n   elif self.entry_variant=="B_CONFIRM_A":\n    if apass and B.direction==A.direction and B.direction!=0 and B.score>=.60: selected_edge="A"; d=A.direction\n   else:\n    if apass or bpass:\n     if bpass and (not apass or be>ae): selected_edge="B"; d=B.direction\n     else: selected_edge="A"; d=A.direction'
if old not in s: raise SystemExit('v5 direction target missing')
s=s.replace(old,new)
old='if not d:self.blocked+=1; return\n   inst=self.cache.instrument(self.config.instrument_id); q=inst.make_qty(self.config.trade_size); side=OrderSide.BUY if d>0 else OrderSide.SELL; self.pending=True; self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return'
new='if not d:self.blocked+=1; self.entry_gate_blocked+=1; return\n   self.pending_edge=selected_edge; self.edge_entries[selected_edge]=self.edge_entries.get(selected_edge,0)+1\n   inst=self.cache.instrument(self.config.instrument_id); q=inst.make_qty(self.config.trade_size); side=OrderSide.BUY if d>0 else OrderSide.SELL; self.pending=True; self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return'
if old not in s: raise SystemExit('v5 submit target missing')
s=s.replace(old,new)
old='ea=A.score*A.expected_move-A.cost if A.direction else -1e9; eb=B.score*B.expected_move-B.cost if B.direction else -1e9; best=A if ea>=eb else B\n   a_rev=((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0))\n   b_rev=((self.direction>0 and B.direction<0) or (self.direction<0 and B.direction>0))\n   ab_rev=(best.direction!=0 and best.direction==-self.direction and max(ea,eb)>0)\n   normal_reverse=a_rev if self.exit_mode=="A_EXIT" else b_rev if self.exit_mode=="B_EXIT" else ab_rev if self.exit_mode=="AB_EXIT" else False'
new='ae=A.score*A.expected_move-spread if A.direction else -1e9; be=B.score*B.expected_move-spread if B.direction else -1e9; best=A if ae>=be else B\n   a_rev=((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0))\n   b_rev=((self.direction>0 and B.direction<0) or (self.direction<0 and B.direction>0))\n   ab_rev=(best.direction!=0 and best.direction==-self.direction and max(ae,be)>0)\n   if self.entry_variant=="OWNED_EXIT": normal_reverse=(a_rev if self.active_edge=="A" else b_rev if self.active_edge=="B" else False)\n   else: normal_reverse=b_rev'
if old not in s: raise SystemExit('v5 exit target missing')
s=s.replace(old,new)
old='self.opened+=1; self.pending=False; self.entry_px=self.f(e.avg_px_open)'
new='self.opened+=1; self.pending=False; self.entry_px=self.f(e.avg_px_open); self.active_edge=self.pending_edge'
if old not in s: raise SystemExit('v5 opened target missing')
s=s.replace(old,new)
old='self.direction=0; self.closed+=1; self.pending=False; self.entry_px=None; self.debt=0.; self.debt_hist.clear(); self.trades.append({"pnl":parse_money(e.realized_pnl),"ts_closed":int(e.ts_closed or e.ts_event)})'
new='self.direction=0; self.closed+=1; self.pending=False; self.entry_px=None; self.debt=0.; self.debt_hist.clear(); self.active_edge=None; self.pending_edge=None; self.trades.append({"pnl":parse_money(e.realized_pnl),"ts_closed":int(e.ts_closed or e.ts_event)})'
if old not in s: raise SystemExit('v5 closed target missing')
s=s.replace(old,new)
old='"exit_mode":s.exit_mode,"interventions":s.interventions'
new='"exit_mode":s.exit_mode,"entry_variant":s.entry_variant,"edge_entries":s.edge_entries,"entry_gate_blocked":s.entry_gate_blocked,"interventions":s.interventions'
if old not in s: raise SystemExit('v5 metrics target missing')
s=s.replace(old,new)
p.write_text(s)
compile(s,str(p),'exec')
print('AE_MATH_V5_OWNERSHIP_PATCH_OK')
