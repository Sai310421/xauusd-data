from pathlib import Path
p=Path('research/ae_highn_2edge_math_raw6x3_bt.py')
s=p.read_text()
# Harmonic D ratio: use AD/XA instead of XD/XA proxy, and expose reversible direction diagnostic.
old='rb=ab/xa; dxa=abs(d-x)/xa; cdbc=cd/bc; tol=.03'
new='rb=ab/xa; adxa=abs(d-a)/xa; cdbc=cd/bc; tol=.03'
if old not in s: raise SystemExit('v6 ratio target missing')
s=s.replace(old,new)
s=s.replace('abs(dxa-.786)','abs(adxa-.786)')
s=s.replace('abs(dxa-1.272)','abs(adxa-1.272)')
s=s.replace('abs(dxa-.886)','abs(adxa-.886)')
s=s.replace('{"rb":rb,"dxa":dxa}','{"rb":rb,"adxa":adxa}')
# Direction diagnostic variants after v5 computes d / selected_edge.
old='selected_edge=None; d=0\n   if self.entry_variant=="B_ONLY":'
new='selected_edge=None; d=0\n   if self.entry_variant=="B_ONLY":'
if old not in s: raise SystemExit('v6 direction anchor missing')
# no structural change at anchor; add inversion immediately before flat-position gate
old='if self.cache.positions_open_count(instrument_id=self.config.instrument_id) == 0 and not self.pending:'
new='diag=os.getenv("DIRECTION_DIAG","NATIVE").upper()\n  if diag=="INVERT_ALL" and d!=0: d=-d\n  elif diag=="INVERT_B" and selected_edge=="B" and d!=0: d=-d\n  elif diag=="INVERT_A" and selected_edge=="A" and d!=0: d=-d\n  if self.cache.positions_open_count(instrument_id=self.config.instrument_id) == 0 and not self.pending:'
if old not in s: raise SystemExit('v6 flat gate target missing')
s=s.replace(old,new)
# Bias diagnostic: allow contrarian harmonic bias instead of trend-following bonus.
old='score=clamp(base+.25*max(0,clamp(bias*dr)),0,1); dr=dr if score>=.60 else 0'
new='bias_mode=os.getenv("HARM_BIAS_MODE","CONTRARIAN").upper(); bterm=(-bias*dr) if bias_mode=="CONTRARIAN" else (bias*dr); score=clamp(base+.25*max(0,clamp(bterm)),0,1); dr=dr if score>=.60 else 0'
if old not in s: raise SystemExit('v6 bias target missing')
s=s.replace(old,new)
p.write_text(s)
compile(s,str(p),'exec')
print('AE_MATH_V6_DIRECTION_PATCH_OK')
