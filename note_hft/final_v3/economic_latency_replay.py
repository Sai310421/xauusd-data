#!/usr/bin/env python3
import argparse,json,lzma,struct,bisect
from pathlib import Path
from reconstructed_alpha import ReconstructedDirectionalAlpha
REC=struct.Struct('>3i2f'); SCALE=1000.0

def metrics(pnls):
    n=len(pnls); wins=sum(x>0 for x in pnls); gp=sum(x for x in pnls if x>0); gl=-sum(x for x in pnls if x<0)
    eq=peak=0.0; maxdd_abs=0.0
    for x in pnls:
        eq+=x; peak=max(peak,eq); maxdd_abs=max(maxdd_abs,peak-eq)
    # DD percent requires capital/notional convention. Report normalized DD against gross-profit scale too, but do not mislabel it account DD.
    return {'N':n,'WR_pct':100*wins/n if n else None,'PF':gp/gl if gl>0 else None,'net_price_pnl':sum(pnls),'gross_profit':gp,'gross_loss':gl,'max_drawdown_price':maxdd_abs,'expectancy_price':sum(pnls)/n if n else None}

def load_ticks(root):
    out=[]
    files=sorted(Path(root).rglob('*h_ticks.bi5'))
    hour=0
    for f in files:
        try: dec=lzma.decompress(f.read_bytes())
        except: hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
            out.append((hour*3600000+ms,bid_i/SCALE,ask_i/SCALE))
        hour+=1
    return out,files

def run(ticks,entry_ms,position_ms,close_ms,window=5):
    alpha=ReconstructedDirectionalAlpha(window=window); last_create=-10**30; ts=[x[0] for x in ticks]; pnls=[]; buy=sell=0; holds=[]
    for idx,(t,bid,ask) in enumerate(ticks):
        r=alpha.update(ask,bid); side=r.signal
        if not side or t-last_create<3000: continue
        # signal/order at t; fill on first observed quote after entry latency
        ei=bisect.bisect_left(ts,t+entry_ms,idx)
        if ei>=len(ticks): break
        et,eb,ea=ticks[ei]
        # public core closes after position is observed, then close order latency applies
        target=et+position_ms+close_ms
        xi=bisect.bisect_left(ts,target,ei+1)
        if xi>=len(ticks): break
        xt,xb,xa=ticks[xi]
        if side>0:
            pnl=xb-ea; buy+=1
        else:
            pnl=eb-xa; sell+=1
        pnls.append(pnl); holds.append(xt-et); last_create=t
    m=metrics(pnls); m.update({'BUY':buy,'SELL':sell,'entry_latency_ms':entry_ms,'position_snapshot_ms':position_ms,'close_latency_ms':close_ms,'avg_holding_ms':sum(holds)/len(holds) if holds else None})
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='economic.json'); a=ap.parse_args()
    ticks,files=load_ticks(a.root)
    profiles=[('FAST_10_10_10',10,10,10),('NORMAL_20_20_20',20,20,20),('NORMAL_20_50_20',20,50,20),('STRESS_80_80_80',80,80,80),('TAIL_250_250_250',250,250,250)]
    res={name:run(ticks,e,p,c) for name,e,p,c in profiles}
    out={'status':'REFERENCE_FEED_ECONOMIC_PROXY','warning':'Dukascopy has nonzero spread and is reference feed, not the original zero-spread execution broker. WR/PF are sensitivity/proxy results, not final NOTE-HFT broker parity. Account MaxDD% is intentionally not fabricated without original capital/notional convention.','ticks':len(ticks),'bi5_files':len(files),'baseline':{'WR_pct':72.71,'PF':1.74,'MaxDD_pct':3.97,'N':176483},'profiles':res}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
