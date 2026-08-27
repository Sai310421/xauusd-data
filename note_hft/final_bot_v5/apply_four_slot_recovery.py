#!/usr/bin/env python3
"""Apply the recovered four masked expressions to the user's original NOTE-HFT source.

Design rule: preserve the original source as the baseline. This patcher only:
1) restores the missing rolling ask/bid history maintenance required by the visible startup gate,
2) replaces the four 00000 masks,
3) initializes the recovered a/b state so the original downstream code can run.

It intentionally does not alter order flow, 3s permit, spread gate, 0.1s sleeps,
lot sizing, margin gate, ZMQ topology, or close-first behavior.
"""
from pathlib import Path
import sys, hashlib, json

MASK = '''                if 00000 and 00000:\n                    self.a_cond_num=00000\n                    self.b_cond_num=00000\n                else:\n                    self.a_cond_num=0\n                    self.b_cond_num=0'''

RECOVERED = '''                # Recovered four masked expressions (v5)\n                # SLOT1: enough ask history\n                # SLOT2: enough bid history\n                # SLOT3: ask-side signed move, oriented for the frozen BUY/SELL predicates\n                # SLOT4: bid-side signed move, oriented for the frozen BUY/SELL predicates\n                if len(self.ask_list) >= 5 and len(self.bid_list) >= 5:\n                    self.a_cond_num = self.ask_list[0] - self.ask_list[-1]\n                    self.b_cond_num = self.bid_list[-1] - self.bid_list[0]\n                else:\n                    self.a_cond_num=0\n                    self.b_cond_num=0'''

ANCHOR = '''                self.udp_sp=self.udp_ask-self.udp_bid\n                self.udp_mid=(self.udp_ask+self.udp_bid)/2\n'''

HISTORY = '''                self.udp_sp=self.udp_ask-self.udp_bid\n                self.udp_mid=(self.udp_ask+self.udp_bid)/2\n\n                # Restore the rolling history implied by the visible 5-sample startup gate.\n                self.ask_list.append(self.udp_ask)\n                self.bid_list.append(self.udp_bid)\n                if len(self.ask_list) > 5:\n                    del self.ask_list[:-5]\n                if len(self.bid_list) > 5:\n                    del self.bid_list[:-5]\n'''

INIT_ANCHOR = '''        self.udp_ask,self.udp_bid=None,None\n'''
INIT_REPLACEMENT = '''        self.udp_ask,self.udp_bid=None,None\n        self.a_cond_num=Decimal('0')\n        self.b_cond_num=Decimal('0')\n'''

def one(src, old, new, label):
    n=src.count(old)
    if n!=1:
        raise RuntimeError(f'{label}: expected one exact match, found {n}; refusing unsafe patch')
    return src.replace(old,new,1)

def main():
    if len(sys.argv)!=3:
        raise SystemExit('usage: python apply_four_slot_recovery.py ORIGINAL.py COMPLETE.py')
    inp,out=map(Path,sys.argv[1:])
    src=inp.read_text(encoding='utf-8')
    before=hashlib.sha256(src.encode()).hexdigest()
    src=one(src,INIT_ANCHOR,INIT_REPLACEMENT,'state initialization')
    src=one(src,ANCHOR,HISTORY,'rolling history restoration')
    src=one(src,MASK,RECOVERED,'four masked slots')
    out.write_text(src,encoding='utf-8')
    rep={
      'original_sha256':before,
      'completed_sha256':hashlib.sha256(src.encode()).hexdigest(),
      'changes_only':['initialize a_cond_num/b_cond_num','restore 5-sample ask/bid rolling history','recover four masked expressions'],
      'recovered_slots':{
        'slot1':'len(self.ask_list) >= 5',
        'slot2':'len(self.bid_list) >= 5',
        'slot3':'self.ask_list[0] - self.ask_list[-1]',
        'slot4':'self.bid_list[-1] - self.bid_list[0]'
      },
      'untouched':['order flow','spread gate','3-second create permit','0.1-second sleeps','lot sizing','margin gate','ZMQ ports/topology','close-first execution']
    }
    Path(str(out)+'.recovery.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rep,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
