from decimal import Decimal
from reconstructed_alpha import ReconstructedDirectionalAlpha

def test_up_is_buy():
    e=ReconstructedDirectionalAlpha(window=5)
    out=None
    for i in range(5): out=e.update(100+i,99+i)
    assert out.ready and out.signal==1 and out.a_cond_num<0 and out.b_cond_num>0

def test_down_is_sell():
    e=ReconstructedDirectionalAlpha(window=5)
    out=None
    for i in range(5): out=e.update(104-i,103-i)
    assert out.ready and out.signal==-1 and out.a_cond_num>0 and out.b_cond_num<0

def test_flat_none():
    e=ReconstructedDirectionalAlpha(window=5)
    out=None
    for _ in range(5): out=e.update(100,99)
    assert out.ready and out.signal==0

def test_default_warmup_is_five():
    e=ReconstructedDirectionalAlpha()
    for _ in range(4): assert not e.update(100,99).ready
    assert e.update(101,100).ready
