from pathlib import Path

p = Path('research/ae_harmonic_sequence_runner.py')
s = p.read_text()
old = 'self.h=HFT(); self.hm=Harm(); self.swing=SwingSeq(depth=2); self.bs=deque(maxlen=32);'
new = 'self.h=HFT(); self.hm=Harm(); self.ctl=Controller(); self.swing=SwingSeq(depth=2); self.bs=deque(maxlen=32);'
if old not in s:
    raise SystemExit('harmonic sequence init target missing')
s = s.replace(old, new, 1)
p.write_text(s)
compile(s, str(p), 'exec')
print('AE_HARMONIC_SEQUENCE_HOTFIX_OK')
