from pathlib import Path

p = Path('research/ae_harmonic_sequence_runner.py')
s = p.read_text()

old = '''  if mode=="AB100D" and act in (Action.EXIT,Action.REDUCE,Action.HEDGE): self.close_all_positions(self.config.instrument_id)\n  elif mode!="AB100D" and ((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0)): self.close_all_positions(self.config.instrument_id)\n'''
new = '''  if mode=="AB100D" and act in (Action.EXIT,Action.REDUCE,Action.HEDGE):\n   self.close_all_positions(self.config.instrument_id)\n  elif mode=="B":\n   # B/Harmonic must complete its own state transition. Never use A/HFT polarity as B exit.\n   pid=B.meta.get("pattern_id") if B.meta else None\n   fresh=(pid is not None and pid!=self.last_b_entry_id and B.direction!=0)\n   opposite=fresh and ((self.direction>0 and B.direction<0) or (self.direction<0 and B.direction>0))\n   if opposite:\n    self.close_all_positions(self.config.instrument_id)\n  elif mode!="AB100D" and ((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0)):\n   self.close_all_positions(self.config.instrument_id)\n'''

if old not in s:
    raise SystemExit('B exit target missing')
s = s.replace(old, new, 1)
p.write_text(s)
compile(s, str(p), 'exec')
print('AE_HARMONIC_SEQUENCE_BEXIT_PATCH_OK')
