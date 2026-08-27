#!/usr/bin/env python3
import argparse, json, lzma, struct, re
from pathlib import Path
from reconstructed_alpha import ReconstructedDirectionalAlpha
REC=struct.Struct('>3i2f'); SCALE=1000.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--window',type=int,default=5); ap.add_argument('-o','--out',default='bi5_analysis.json'); a=ap.parse_args()
    files=sorted(Path(a.root).rglob('*h_ticks.bi5'))
    alpha=ReconstructedDirectionalAlpha(window=a.window)
    ticks=zero=rb=rs=pb=ps=0; last_abs_ms=-10**30; hour_index=0; bad=0
    for f in files:
        try: dec=lzma.decompress(f.read_bytes())
        except Exception: bad+=1; continue
        # chronology is provided by sorted YYYY/MM/DD/HH tree; use synthetic absolute ms preserving order/gaps at least hourly
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i); ticks+=1
            ask=ask_i/SCALE; bid=bid_i/SCALE; zero+=int(ask==bid)
            r=alpha.update(ask,bid); sig=r.signal; rb+=int(sig==1); rs+=int(sig==-1)
            abs_ms=hour_index*3600000+ms
            if sig and abs_ms-last_abs_ms>=3000:
                pb+=int(sig==1); ps+=int(sig==-1); last_abs_ms=abs_ms
        hour_index+=1
    out={'status':'OK' if ticks else 'NO_DATA','bi5_files':len(files),'bad_files':bad,'ticks':ticks,'zero_spread_ticks':zero,'zero_spread_ratio':zero/ticks if ticks else None,'raw_signal_buy':rb,'raw_signal_sell':rs,'raw_signal_n':rb+rs,'permit3s_buy':pb,'permit3s_sell':ps,'permit3s_n':pb+ps,'window':a.window,'baseline':{'N':176483,'BUY':88223,'SELL':88260,'WR':72.71,'PF':1.74,'MaxDD':3.97}}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
