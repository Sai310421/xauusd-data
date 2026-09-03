from pathlib import Path
import os

p=Path('research/ae_original_mirror_runner.py')
s=p.read_text()

# Execution-only Raw L1 compatibility patch. Prices/timestamps and strategy logic remain original.
s=s.replace('import argparse,json,math','import argparse,json,math,os')
s=s.replace('from nautilus_trader.model import BarType,Money,Venue','from nautilus_trader.model import BarType,Money,Venue\nfrom nautilus_trader.model.enums import BookType')
s=s.replace('from nautilus_trader.model.identifiers import InstrumentId','from nautilus_trader.model.identifiers import InstrumentId\nfrom nautilus_trader.model.objects import Quantity')
s=s.replace('cat.query_quote_ticks(identifiers=[inst.id.value])','cat.quote_ticks(instrument_ids=[inst.id.value])')
s=s.replace('e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))','e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))')

run_old='def run(inst,ticks,sym,tf,mode,days):\n cfg='
run_new='def run(inst,ticks,sym,tf,mode,days):\n zb=sum(1 for t in ticks if t.bid_size.as_double()<=0); za=sum(1 for t in ticks if t.ask_size.as_double()<=0)\n if zb or za:\n  one=Quantity.from_int(1)\n  ticks=[QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=t.bid_size if t.bid_size.as_double()>0 else one,ask_size=t.ask_size if t.ask_size.as_double()>0 else one,ts_event=t.ts_event,ts_init=t.ts_init) for t in ticks]\n cfg='
if run_old not in s: raise SystemExit('run header target missing')
s=s.replace(run_old,run_new)

# Stable fill accounting only; no strategy decision changes.
s=s.replace('self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.', 'self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.; self.trades=[]; self.fills=0; self.rejects=0; self.opened=0; self.closed=0; self.pending=False; self.reject_reasons={}; self.mirror=(os.getenv("ORIGINAL_DIRECTION","ORIGINAL").upper()=="MIRROR")')
s=s.replace('if self.portfolio.is_net_flat(self.config.instrument_id):', 'if self.cache.positions_open_count(instrument_id=self.config.instrument_id) == 0 and not self.pending:')
s=s.replace('self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return', 'self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; self.pending=True; return')

ev_old=' def on_position_closed(self,e): self.direction=0'
ev_new=' def on_order_filled(self,e): self.fills+=1; self.pending=False\n def _reject(self,e):\n  self.rejects+=1; self.pending=False\n  r=str(getattr(e,"reason","UNKNOWN")); self.reject_reasons[r]=self.reject_reasons.get(r,0)+1\n def on_order_rejected(self,e): self._reject(e)\n def on_order_denied(self,e): self._reject(e)\n def on_order_canceled(self,e): self.pending=False\n def on_position_opened(self,e): self.opened+=1; self.pending=False\n def on_position_closed(self,e):\n  self.direction=0; self.closed+=1; self.pending=False\n  self.trades.append({"pnl":parse_money(e.realized_pnl),"ts_closed":int(e.ts_closed or e.ts_event)})'
if ev_old not in s: raise SystemExit('event target missing')
s=s.replace(ev_old,ev_new)

# The ONLY strategy-logic experimental change: flip A and B signal directions together.
old='bid,ask=self.f(t.bid_price),self.f(t.ask_price); A=self.h.on(Tick(float(t.ts_event),bid,ask,self.f(t.bid_size),self.f(t.ask_size))); B=self.B; act,val,ds,p,inter,adv=self.ctl.decide(A,B,self.debt,0,.8,-self.debt); self.actions[act.value]=self.actions.get(act.value,0)+1'
new='bid,ask=self.f(t.bid_price),self.f(t.ask_price); A=self.h.on(Tick(float(t.ts_event),bid,ask,self.f(t.bid_size),self.f(t.ask_size))); B=self.B\n  if self.mirror:\n   A=Sig(A.name,-A.direction,A.score,A.expected_move,A.cost,A.meta); B=Sig(B.name,-B.direction,B.score,B.expected_move,B.cost,B.meta)\n  act,val,ds,p,inter,adv=self.ctl.decide(A,B,self.debt,0,.8,-self.debt); self.actions[act.value]=self.actions.get(act.value,0)+1'
if old not in s: raise SystemExit('signal mirror target missing')
s=s.replace(old,new)

old='e.run(); trades=extract(e.generate_positions_report(),sym,tf); m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
new='e.run(); trades=[{"symbol":sym,"tf":tf,**x} for x in s.trades]; m=metr(trades,days=days); m.update({"original_direction":"MIRROR" if s.mirror else "ORIGINAL","signals":s.entries,"blocked":s.blocked,"fills":s.fills,"rejected":s.rejects,"opened":s.opened,"closed":s.closed,"reject_reasons":s.reject_reasons,"zero_bid_sizes":zb,"zero_ask_sizes":za,"liquidity_size_floor":1,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m'
if old not in s: raise SystemExit('metrics target missing')
s=s.replace(old,new)

p.write_text(s)
compile(s,str(p),'exec')
print('AE_ORIGINAL_MIRROR_PATCH_OK')
