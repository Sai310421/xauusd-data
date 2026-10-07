from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

VARIANTS=("CORE","MACRO","PDA_VOL","VIDEO_OR")

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def nsdt(x): return pd.Timestamp(int(x),unit='ns',tz='UTC')
def metrics(rs,risk=.35):
    eq=pk=100.;dd=0.;gp=gl=0.
    for r in rs:
        if r>0: gp+=r
        else: gl-=r
        eq*=max(.0001,1+risk*r/100);pk=max(pk,eq);dd=max(dd,100*(pk-eq)/pk)
    return {'N':len(rs),'wins':sum(r>0 for r in rs),'WR_pct':100*sum(r>0 for r in rs)/len(rs) if rs else 0.,
            'PF_R':gp/gl if gl else (math.inf if gp else 0.),'sum_R':sum(rs),'Return_pct':eq-100,'MaxDD_pct':dd}

class Cfg(StrategyConfig,frozen=True):
    instrument_id: object

class V18(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.bars=[];self.cur=None;self.phase=0;self.age=0;self.di=0
        self.acc_hi=self.acc_lo=None;self.sweep=None;self.cisd=None;self.confirm_i=-1;self.route=''
        self.positions={v:None for v in VARIANTS};self.rs={v:[] for v in VARIANTS};self.trades=[]
        self.stage=defaultdict(lambda:defaultdict(int));self.events=[]
        self.tick_counts=[];self.setup_ctx={};self.entry_count=0;self.last_entry_i=-1
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def route_at(self,t):
        j=t.tz_convert('Asia/Tokyo');m=j.hour*60+j.minute
        if 540<=m<600:return 'ASIA'
        if 960<=m<1020:return 'LONDON'
        if 1380<=m<1440:return 'NY'
        return None
    def macro(self,route,t):
        j=t.tz_convert('Asia/Tokyo');m=j.minute
        return m<30
    def build_range(self,route,t):
        j=t.tz_convert('Asia/Tokyo');day=j.normalize();xs=[]
        for b in self.bars[:-1]:
            z=b['t'].tz_convert('Asia/Tokyo');m=z.hour*60+z.minute
            if route=='LONDON' and z.normalize()==day and 540<=m<960: xs.append(b)
            elif route=='NY' and z.normalize()==day and 960<=m<1380: xs.append(b)
            elif route=='ASIA' and ((z.normalize()==day-pd.Timedelta(days=1) and m>=1380) or (z.normalize()==day and m<360)): xs.append(b)
        return None if not xs else (max(x['h'] for x in xs),min(x['l'] for x in xs))
    def atr(self,n=14):
        if len(self.bars)<n+2:return None
        x=self.bars[-n:];trs=[]
        for k,b in enumerate(x):
            prev=self.bars[len(self.bars)-n+k-1]['c']
            trs.append(max(b['h']-b['l'],abs(b['h']-prev),abs(b['l']-prev)))
        return sum(trs)/len(trs)
    def delivery(self,b,di):return b['c']>b['o'] if di<0 else b['c']<b['o']
    def cisd_origin(self,i,di):
        j=i
        if not self.delivery(self.bars[j],di):j-=1
        if j<1 or not self.delivery(self.bars[j],di):return None
        oldest=j
        for k in range(j-1,max(-1,i-8),-1):
            if not self.delivery(self.bars[k],di):break
            oldest=k
        return self.bars[oldest]['o']
    def htf_pda(self,i,di,px):
        # Causal M30 proxy from two completed M15 bars; use last 24 M30-equivalent blocks.
        if i<50:return False
        pairs=[]
        end=i-1
        k=end-(end%2)
        for z in range(max(1,k-48),k+1,2):
            chunk=self.bars[z:z+2]
            if len(chunk)==2:pairs.append((max(x['h'] for x in chunk),min(x['l'] for x in chunk)))
        if len(pairs)<24:return False
        w=pairs[-24:];mid=(max(x[0] for x in w)+min(x[1] for x in w))/2
        return px>=mid if di<0 else px<=mid
    def volume_influx(self):
        # QuoteTick-count proxy only; raw source has no trusted exchange volume.
        if len(self.tick_counts)<21:return False
        prev=sorted(self.tick_counts[-21:-1]);med=prev[len(prev)//2]
        return med>0 and self.tick_counts[-1]>=1.15*med
    def m30_fvg_targets(self,i,di,entry,look=40):
        out=[]
        # Build causal M30 candles from completed M15 pairs only.
        end=i-1
        k=end-(end%2)
        pairs=[]
        for z in range(max(0,k-2*look-4),k+1,2):
            ch=self.bars[z:z+2]
            if len(ch)==2:
                pairs.append({'h':max(x['h'] for x in ch),'l':min(x['l'] for x in ch)})
        for n in range(2,len(pairs)):
            if pairs[n]['l']>pairs[n-2]['h']:
                lo=pairs[n-2]['h'];hi=pairs[n]['l']
                if di>0 and lo>entry:out.extend([lo,hi])
            if pairs[n]['h']<pairs[n-2]['l']:
                lo=pairs[n]['h'];hi=pairs[n-2]['l']
                if di<0 and hi<entry:out.extend([hi,lo])
        return out
    def target(self,i,entry,sl,di,acc_hi,acc_lo):
        risk=abs(entry-sl);cs=[]
        for k in range(max(1,i-24),i):
            if di<0 and self.bars[k]['l']<self.bars[k-1]['l'] and self.bars[k]['l']<self.bars[min(k+1,len(self.bars)-1)]['l'] and self.bars[k]['l']<entry:cs.append(self.bars[k]['l'])
            if di>0 and self.bars[k]['h']>self.bars[k-1]['h'] and self.bars[k]['h']>self.bars[min(k+1,len(self.bars)-1)]['h'] and self.bars[k]['h']>entry:cs.append(self.bars[k]['h'])
        cs.extend(self.m30_fvg_targets(i,di,entry));cs.append(acc_lo if di<0 else acc_hi);good=[]
        for x in cs:
            rew=(entry-x) if di<0 else (x-entry);rr=rew/risk if risk else 0
            if rew>0 and rr>=.8:good.append((rew,x,rr))
        return min(good) if good else None
    def passv(self,v,ctx):
        if v=='CORE':return True
        if v=='MACRO':return ctx['macro']
        if v=='PDA_VOL':return ctx['pda'] and ctx['vol']
        if v=='VIDEO_OR':return ctx['macro'] or (ctx['pda'] and ctx['vol'])
        return False
    def reset(self):
        self.phase=0;self.age=0;self.di=0;self.acc_hi=self.acc_lo=None;self.sweep=None;self.cisd=None;self.confirm_i=-1;self.route='';self.setup_ctx={};self.entry_count=0;self.last_entry_i=-1
    def finish_bar(self):
        if self.cur is None:return
        self.bars.append(self.cur)
        self.tick_counts.append(self.cur['ticks'])
        self.process_bar(len(self.bars)-1);self.cur=None
    def process_bar(self,i):
        b=self.bars[i];a=self.atr()
        if a is None:return
        if self.phase==0:
            r=self.route_at(b['t'])
            if not r:return
            z=self.build_range(r,b['t'])
            if not z:return
            self.route=r;self.acc_hi,self.acc_lo=z;self.phase=1;self.age=0;self.stage[r]['armed']+=1
            return
        self.age+=1
        if self.phase==1:
            up=b['h']>self.acc_hi+a*.03 and b['c']<self.acc_hi
            dn=b['l']<self.acc_lo-a*.03 and b['c']>self.acc_lo
            if up or dn:
                self.di=-1 if up else 1;self.sweep=b['h'] if up else b['l'];self.cisd=self.cisd_origin(i,self.di)
                self.stage[self.route]['sweep']+=1
                if self.cisd is None:self.reset();return
                self.stage[self.route]['cisd_candle']+=1
                self.setup_ctx={'macro':self.macro(self.route,b['t']),'pda':self.htf_pda(i,self.di,b['c']),'vol':False}
                self.phase=2;self.age=0
            elif self.age>8:self.reset()
            return
        if self.phase==2:
            # While waiting for confirmation, a deeper manipulation/sweep can form.
            # Treat it as the same AMD event and refresh the CISD candle from the newest delivery sequence
            # instead of discarding the later video-valid sequence.
            resweep=False
            if self.di<0 and b['h']>self.sweep:
                self.sweep=b['h'];resweep=True
            elif self.di>0 and b['l']<self.sweep:
                self.sweep=b['l'];resweep=True
            if resweep:
                level=self.cisd_origin(i,self.di)
                if level is not None:
                    self.cisd=level
                    self.setup_ctx={'macro':self.macro(self.route,b['t']),'pda':self.htf_pda(i,self.di,b['c']),'vol':False}
                    self.stage[self.route]['cisd_refresh']+=1
                    self.age=0
            ok=b['c']<self.cisd if self.di<0 else b['c']>self.cisd
            if ok:
                self.confirm_i=i;self.setup_ctx['vol']=self.volume_influx();self.phase=3;self.age=0
                self.stage[self.route]['confirmed']+=1
            elif self.age>8:self.reset()
            return
        if self.phase==3:
            touch=b['h']>=self.cisd if self.di<0 else b['l']<=self.cisd
            hold=b['c']<=self.cisd if self.di<0 else b['c']>=self.cisd
            # Video-valid repeated CISD retests are separate entry opportunities.
            # Never duplicate the same M15 bar; cap at 3 entries per confirmed structure.
            if touch and hold and i>self.confirm_i and i!=self.last_entry_i:
                self.stage[self.route]['entry_signal']+=1
                p={'pending':True,'di':self.di,'sl_ref':self.sweep,'atr':a,'route':self.route,'signal':str(b['t']),
                   'acc_hi':self.acc_hi,'acc_lo':self.acc_lo,'ctx':dict(self.setup_ctx),'cisd':self.cisd}
                opened=False
                for v in VARIANTS:
                    if self.positions[v] is None and self.passv(v,p['ctx']):
                        self.positions[v]=dict(p,variant=v);opened=True
                if opened:
                    self.entry_count+=1;self.last_entry_i=i;self.age=0
                if self.entry_count>=3:self.reset()
            elif self.age>8:self.reset()
    def on_quote_tick(self,t):
        bid,ask=f(t.bid_price),f(t.ask_price);ts=nsdt(t.ts_event)
        for v in VARIANTS:
            p=self.positions[v]
            if p and not p.get('pending'):
                mark=bid if p['di']>0 else ask;hit=None
                if p['di']>0:
                    if mark<=p['sl']:hit=-1.
                    elif mark>=p['tp']:hit=p['rr']
                else:
                    if mark>=p['sl']:hit=-1.
                    elif mark<=p['tp']:hit=p['rr']
                if hit is not None:
                    self.rs[v].append(hit);self.trades.append({**p,'exit':str(ts),'R':hit});self.positions[v]=None
            p=self.positions[v]
            if p and p.get('pending'):
                entry=ask if p['di']>0 else bid
                sl=p['sl_ref']-p['atr']*.05 if p['di']>0 else p['sl_ref']+p['atr']*.05
                spread=ask-bid;risk=abs(entry-sl)
                if risk<=max(spread*2.0,1e-9):
                    self.positions[v]=None;continue
                z=self.target(len(self.bars)-1,entry,sl,p['di'],p['acc_hi'],p['acc_lo'])
                if z is None:
                    self.positions[v]=None;continue
                _,tp,rr=z
                self.stage[p['route']]['target_ok']+=1
                self.positions[v]={**p,'pending':False,'entry':entry,'sl':sl,'tp':tp,'rr':rr,'entry_time':str(ts),'spread':spread}
        bucket=ts.floor('15min');mid=(bid+ask)/2
        if self.cur is None:self.cur={'t':bucket,'o':mid,'h':mid,'l':mid,'c':mid,'ticks':1}
        elif bucket!=self.cur['t']:
            self.finish_bar();self.cur={'t':bucket,'o':mid,'h':mid,'l':mid,'c':mid,'ticks':1}
        else:
            self.cur['h']=max(self.cur['h'],mid);self.cur['l']=min(self.cur['l'],mid);self.cur['c']=mid;self.cur['ticks']+=1
    def on_stop(self):self.finish_bar()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);ap.add_argument('--start');ap.add_argument('--end');a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value],start=a.start,end=a.end);assert raw
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(raw);st=V18(Cfg(instrument_id=inst.id));eng.add_strategy(st);eng.run();eng.end()
    p=Path(a.out);p.mkdir(parents=True,exist_ok=True)
    rows=[]
    for v in VARIANTS:rows.append({'variant':v,**metrics(st.rs[v])})
    pd.DataFrame(rows).to_csv(p/'variant_kpi.csv',index=False)
    pd.DataFrame(st.trades).to_csv(p/'trades.csv',index=False)
    stages={r:dict(x) for r,x in st.stage.items()}
    (p/'stages.json').write_text(json.dumps(stages,indent=2))
    out={'verification':'AMOS_VIDEO_PARITY_CISD_V2_0_RAW_MULTI_RETEST_M30FVG','raw_ticks':len(raw),'query_start':a.start,'query_end':a.end,
         'video_only':True,'ote_used':False,'volume_note':'Volume Influx uses raw QuoteTick count proxy because upstream quote feed has no trusted exchange volume',
         'variants':rows,'stages':stages}
    (p/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
