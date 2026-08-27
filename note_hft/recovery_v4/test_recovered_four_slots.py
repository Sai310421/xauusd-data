from decimal import Decimal as D
from recovered_four_slots import FourSlotRecovery


def feed(seq):
    e=FourSlotRecovery()
    out=None
    for ask,bid in seq:
        out=e.push(ask,bid)
    return out

# Before five samples, both history guards are false and no signal can exist.
e=FourSlotRecovery()
for i in range(4):
    r=e.push(D('100.02')+D(i)/100, D('100.00')+D(i)/100)
    assert not (r.slot1 and r.slot2)
    assert r.signal==0

# Coherent upward shift => a<0,b>0 => BUY.
r=feed([('100.02','100.00'),('100.03','100.01'),('100.04','100.02'),('100.05','100.03'),('100.06','100.04')])
assert r.slot1 and r.slot2
assert r.a_cond_num < 0 and r.b_cond_num > 0 and r.signal==1

# Coherent downward shift => a>0,b<0 => SELL.
r=feed([('100.06','100.04'),('100.05','100.03'),('100.04','100.02'),('100.03','100.01'),('100.02','100.00')])
assert r.a_cond_num > 0 and r.b_cond_num < 0 and r.signal==-1

# Freeze => both zero => no trade.
r=feed([('100.02','100.00')]*5)
assert r.a_cond_num==0 and r.b_cond_num==0 and r.signal==0

# Spread widening: ask rises while bid falls => same negative sign => no trade.
r=feed([('100.02','100.00'),('100.03','99.99'),('100.04','99.98'),('100.05','99.97'),('100.06','99.96')])
assert r.a_cond_num < 0 and r.b_cond_num < 0 and r.signal==0

# Spread narrowing: ask falls while bid rises => same positive sign => no trade.
r=feed([('100.06','99.96'),('100.05','99.97'),('100.04','99.98'),('100.03','99.99'),('100.02','100.00')])
assert r.a_cond_num > 0 and r.b_cond_num > 0 and r.signal==0

print('PASS: four-slot source/microstructure invariants')
