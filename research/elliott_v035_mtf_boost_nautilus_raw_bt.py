from __future__ import annotations
import argparse,json,math
from collections import Counter,deque
from dataclasses import dataclass,field
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
    def _qq(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_qq

TFS={'M1':1,'M5':5,'M15':15}
def F(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def exec_ticks(xs):
    one=Quantity.from_int(1);out=[];rep=0
    for t in xs:
        bs,az=F(t.bid_size),F(t.ask_size);rep+=int(bs<=0 or az<=0)
        out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
          bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,
          ts_event=t.ts_event,ts_init=t.ts_init))
    return out,rep

@dataclass
class W:
    state:str='IDLE';d:int=0;o:float=0.;e:float=0.
    fib:list=field(default_factory=lambda:[0.,0.,0.]);touch:list=field(default_factory=lambda:[False]*3)
    entered:list=field(default_factory=lambda:[False]*3);setup:int=0;act:int=0;last:int=0

class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId
    b1:BarType;b5:BarType;b15:BarType
    initial:float=1000.;leverage:int=2000

class V35(Strategy):
    fibv=(.786,.804,.822)
    def __init__(self,c):
        super().__init__(c);self.h={k:deque(maxlen=180) for k in TFS};self.cf={k:0 for k in TFS};self.w={k:W() for k in TFS}
        self.legs=[];self.closed=[];self.baskets=[];self.real=0.;self.peak=c.initial;self.mdd=0.;self.mddp=0.;self.minml=math.inf
        self.maxtot=0.;self.maxnet=0.;self.halt=False;self.hreason=None;self.rec=False;self.rect=0;self.tlat=False;self.tdir=0
        self.bid=self.ask=None;self.now=0;self.oe=Counter();self.sub=0;self.problem=None;self.boost=0;self.we=Counter();self.sc=Counter();self.days=set();self.eqs=[]
    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        for b in (self.config.b1,self.config.b5,self.config.b15): self.subscribe_bars(b)
    def on_order_submitted(self,e):self.oe['submitted']+=1
    def on_order_accepted(self,e):self.oe['accepted']+=1
    def on_order_filled(self,e):self.oe['filled']+=1
    def on_order_denied(self,e):self.oe['denied']+=1;self.problem=str(e)
    def on_order_rejected(self,e):self.oe['rejected']+=1;self.problem=str(e)
    def tfbar(self,b):
        s=str(b.bar_type)
        for k,m in TFS.items():
            if f'-{m}-MINUTE-' in s:return k
    def on_bar(self,b:Bar):
        tf=self.tfbar(b)
        if not tf:return
        r={'t':int(b.ts_event),'o':F(b.open),'h':F(b.high),'l':F(b.low),'c':F(b.close)};q=self.h[tf];q.append(r)
        if len(q)>=2:
            p=q[-2];self.cf[tf]=1 if r['c']>p['h'] else (-1 if r['c']<p['l'] else 0)
    def atr(self,tf,n=14):
        h=list(self.h[tf])
        if len(h)<n+1:return 0.
        z=[]
        for i in range(len(h)-n,len(h)):
            r,p=h[i],h[i-1]['c'];z.append(max(r['h']-r['l'],abs(r['h']-p),abs(r['l']-p)))
        return float(np.mean(z))
    def detect(self,tf):
        h=list(self.h[tf])
        if len(h)<42:return None
        c=h[-1];a=self.atr(tf);ab=np.mean([abs(x['c']-x['o']) for x in h[-21:-1]])
        if a<=0 or ab<=0:return None
        p3=h[-4:-1];bm=c['c']>max(x['h'] for x in p3);sm=c['c']<min(x['l'] for x in p3)
        bd=c['c']>c['o'] and abs(c['c']-c['o'])>=1.6*ab and c['h']-c['l']>=.9*a
        sd=c['c']<c['o'] and abs(c['c']-c['o'])>=1.6*ab and c['h']-c['l']>=.9*a
        bo=so=None
        for off in range(1,7):
            i=len(h)-1-off;b=h[i];old=h[i-12:i]
            lo=min(x['l'] for x in old);hi=max(x['h'] for x in old)
            if bo is None and b['l']<lo and b['c']>lo:bo=b['l']
            if so is None and b['h']>hi and b['c']<hi:so=b['h']
        if bo is not None and bd and bm and c['h']>bo:return 1,bo,c['h'],c['t']
        if so is not None and sd and sm and so>c['l']:return -1,so,c['l'],c['t']
    def arm(self,tf,s):
        d,o,e,t=s;w=self.w[tf];w.state='W1';w.d=d;w.o=o;w.e=e;sp=abs(e-o);w.fib=[e-d*sp*x for x in self.fibv]
        w.touch=[False]*3;w.entered=[False]*3;w.setup=self.now;w.act=0;w.last=t;self.sc[tf]+=1
    def reset(self,tf):self.w[tf]=W()
    def resetall(self):
        for tf in TFS:self.reset(tf)
        self.tlat=False;self.tdir=0
    def inv(self):
        lo=sum(x['q'] for x in self.legs if x['s']>0);sh=sum(x['q'] for x in self.legs if x['s']<0)
        return (lo+sh)/100.,(lo-sh)/100.
    def allow(self,s,lot,cap=.15):
        t,n=self.inv();return t+lot<=min(cap,.15)+1e-12 and abs(n+(lot if s>0 else -lot))<=.12+1e-12
    def submit(self,s,q):
        inst=self.cache.instrument(self.config.instrument_id);o=self.order_factory.market(instrument_id=self.config.instrument_id,
          order_side=OrderSide.BUY if s>0 else OrderSide.SELL,quantity=inst.make_qty(Decimal(str(q))))
        self.sub+=1;self.submit_order(o)
    def openleg(self,s,lot,tag,tf):
        if self.halt or not self.allow(s,lot):return False
        px=self.ask if s>0 else self.bid;q=lot*100
        self.submit(s,q);self.legs.append({'s':s,'q':q,'en':px,'et':self.now,'tag':tag,'tf':tf})
        t,n=self.inv();self.maxtot=max(self.maxtot,t);self.maxnet=max(self.maxnet,abs(n))
        if tf in TFS:self.we[tf]+=1
        if tag=='TRIPLE':self.boost+=1
        return True
    def flt(self):
        return sum(((self.bid if x['s']>0 else self.ask)-x['en'])*x['s']*x['q'] for x in self.legs)
    def eq(self):return self.config.initial+self.real+self.flt()
    def margin(self):
        if not self.legs:return 0.
        return sum(abs(x['q']) for x in self.legs)*((self.bid+self.ask)/2)/self.config.leverage
    def closebasket(self,why):
        if not self.legs:self.resetall();return
        p=0.;rows=[]
        for x in self.legs:
            px=self.bid if x['s']>0 else self.ask;z=(px-x['en'])*x['s']*x['q'];p+=z;self.submit(-x['s'],x['q'])
            y=dict(x);y.update({'ex':px,'xt':self.now,'pnl':z,'reason':why});rows.append(y)
        self.closed+=rows;self.real+=p;self.baskets.append({'xt':self.now,'pnl':p,'reason':why,'legs':len(rows)})
        self.legs=[];self.rec=False;self.rect=0;self.resetall()
    def risk(self):
        e=self.eq();self.peak=max(self.peak,e);dd=self.peak-e;dp=100*dd/max(self.peak,1e-9);self.mdd=max(self.mdd,dd);self.mddp=max(self.mddp,dp)
        m=self.margin()
        if m>0:self.minml=min(self.minml,e/m*100)
        if dp>=6 and not self.halt:self.hreason=f'DD_{dp:.3f}%';self.closebasket('RISK_STOP');self.halt=True
    def exits(self):
        if not self.legs:return False
        p=self.flt()
        if p>=7:self.closebasket('BASKET_TP');return True
        if not self.rec and p<=-15:self.rec=True;self.rect=self.now
        if self.rec and p>=.5 and self.now-self.rect>=30_000_000_000:self.closebasket('RECOVERY_EXIT');return True
        if p<=-500:self.hreason=f'BASKET_{p:.2f}';self.closebasket('BASKET_RISK');self.halt=True;return True
        return False
    def wave(self,tf):
        w=self.w[tf]
        if w.state!='IDLE':
            if self.now-w.setup>240*60*1_000_000_000:self.reset(tf);return
            a=self.atr(tf)
            if a and ((w.d>0 and self.bid<w.o-.2*a) or (w.d<0 and self.ask>w.o+.2*a)):self.reset(tf);return
        w=self.w[tf]
        if w.state=='IDLE':
            s=self.detect(tf)
            if s and s[3]!=w.last:self.arm(tf,s)
            return
        if w.state in ('W1','W2'):
            px=self.bid if w.d>0 else self.ask
            for i,v in enumerate(w.fib):
                if not w.touch[i] and (px<=v if w.d>0 else px>=v):w.touch[i]=True;w.state='W2'
            if w.state!='W2' or self.cf[tf]!=w.d:return
            anye=False
            for i in range(3):
                if w.touch[i] and not w.entered[i] and self.openleg(w.d,.01,f'{tf}_F{i}',tf):w.entered[i]=True;anye=True
            if anye or any(w.entered):w.state='W3';w.act=w.act or self.now
    def adx(self):
        h=list(self.h['M5'])
        if len(h)<120:return None
        hi=np.array([x['h'] for x in h]);lo=np.array([x['l'] for x in h]);cl=np.array([x['c'] for x in h]);n=len(h)
        tr=np.zeros(n);pl=np.zeros(n);mi=np.zeros(n)
        for i in range(1,n):
            tr[i]=max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1]));u=hi[i]-hi[i-1];d=lo[i-1]-lo[i]
            pl[i]=u if u>d and u>0 else 0;mi[i]=d if d>u and d>0 else 0
        p=14;A=np.full(n,np.nan);P=np.full(n,np.nan);M=np.full(n,np.nan);dx=np.full(n,np.nan);ad=np.full(n,np.nan)
        A[p]=tr[1:p+1].sum();P[p]=pl[1:p+1].sum();M[p]=mi[1:p+1].sum()
        for i in range(p,n):
            if i>p:A[i]=A[i-1]-A[i-1]/p+tr[i];P[i]=P[i-1]-P[i-1]/p+pl[i];M[i]=M[i-1]-M[i-1]/p+mi[i]
            if A[i]>0:
                a,b=100*P[i]/A[i],100*M[i]/A[i];dx[i]=100*abs(a-b)/(a+b) if a+b else 0
        st=2*p-1;ad[st]=np.nanmean(dx[p:st+1])
        for i in range(st+1,n):ad[i]=(ad[i-1]*(p-1)+dx[i])/p
        return float(ad[-1]),float(ad[-2]),float(np.mean(tr[-14:])/np.mean(tr[-100:]))
    def boostit(self):
        ws=[self.w[x] for x in ('M1','M5','M15')]
        d=0
        if all(x.state=='W3' and x.d and x.act for x in ws) and len({x.d for x in ws})==1 and max(x.act for x in ws)-min(x.act for x in ws)<=45*60*1_000_000_000:d=ws[0].d
        if not d:self.tlat=False;self.tdir=0;return
        if self.tlat and self.tdir==d:return
        a=self.adx()
        if not a or not(a[0]>23 and a[0]>a[1] and a[2]>1):return
        if self.allow(d,.03) and self.openleg(d,.03,'TRIPLE','BOOST'):self.tlat=True;self.tdir=d
    def on_quote_tick(self,t:QuoteTick):
        self.now=int(t.ts_event);self.bid=F(t.bid_price);self.ask=F(t.ask_price);z=pd.Timestamp(self.now,unit='ns',tz='UTC')
        if z.weekday()<5:self.days.add(z.date().isoformat())
        if self.halt:return
        self.risk()
        if self.halt:return
        if self.exits():self.risk();return
        for tf in ('M1','M5','M15'):self.wave(tf)
        self.boostit();self.risk()
        if not self.eqs or self.now-self.eqs[-1][0]>=300_000_000_000:self.eqs.append((self.now,self.eq()))
    def on_stop(self):
        if self.legs and self.bid is not None:self.closebasket('FORCED_EOT');self.risk()
    def report(self):
        p=np.array([x['pnl'] for x in self.closed]);w=p[p>0];l=p[p<0];pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0)
        b=np.array([x['pnl'] for x in self.baskets]);bw=b[b>0];bl=b[b<0];bpf=float(bw.sum()/abs(bl.sum())) if len(bl) else (math.inf if len(bw) else 0)
        ret=self.real/self.config.initial*100;days=len(self.days)
        return {'N':len(p),'WR_pct':float((p>0).mean()*100) if len(p) else 0,'PF':pf,'RF':self.real/self.mdd if self.mdd else None,
          'Net_USD':self.real,'Return_pct':ret,'MaxFloatingDD_USD':self.mdd,'MaxFloatingDD_pct':self.mddp,
          'MinMarginLevel_pct':None if math.isinf(self.minml) else self.minml,'MaxTotalLots':self.maxtot,'MaxNetLots':self.maxnet,
          'Basket_N':len(b),'Basket_WR_pct':float((b>0).mean()*100) if len(b) else 0,'Basket_PF':bpf,'ExitCounts':dict(Counter(x['reason'] for x in self.baskets)),
          'WaveEntries':dict(self.we),'TripleBoostEntries':self.boost,'Wave1Signals':dict(self.sc),'risk_halted':self.halt,'risk_reason':self.hreason,
          'trading_days':days,'Return_21d_linear_pct':ret*21/days if days else None,'submitted_orders':self.sub,'order_events':dict(self.oe),'native_order_problem':self.problem}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--start',default='2026-02-25T00:00:00Z');ap.add_argument('--end',default='2026-05-26T23:59:59.999Z');a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value],start=a.start,end=a.end)
    if not raw:raise SystemExit('no XAUUSD QuoteTicks')
    ticks,rep=exec_ticks(raw);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks);bs=[BarType.from_str(f'{inst.id.value}-{m}-MINUTE-BID-INTERNAL') for m in (1,5,15)]
    st=V35(Cfg(instrument_id=inst.id,b1=bs[0],b5=bs[1],b15=bs[2]));eng.add_strategy(st);eng.run();eng.end()
    orders=eng.trader.generate_orders_report();fills=eng.trader.generate_order_fills_report();pos=eng.trader.generate_positions_report()
    out={'verification_level':'NAUTILUS_BT_RAW_BIDASK_V035_ELLIOTT_MTF_BOOST','scope':'v0.35 Elliott M1/M5/M15 + Triple Boost + shared basket/risk; legacy Range/CRT/CB/Trend excluded',
      'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(raw),'ohlc_resample_used':False,
      'signal_bars':'Nautilus INTERNAL BID bars from QuoteTicks','execution':'raw Bid/Ask + native market-order fill gate + virtual leg accounting','initial_usd':1000,'leverage':2000,
      'native_orders':0 if orders is None else len(orders),'native_fills':0 if fills is None else len(fills),'native_positions':0 if pos is None else len(pos),'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
      'zero_size_quotes_replaced':rep,**st.report()}
    root=Path('results/elliott-v035-raw')/a.experiment_id;root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(out,indent=2,default=str));pd.DataFrame(st.closed).to_csv(root/'closed_legs.csv',index=False);pd.DataFrame(st.baskets).to_csv(root/'baskets.csv',index=False);pd.DataFrame(st.eqs,columns=['ts_ns','equity']).to_csv(root/'equity_5m.csv',index=False)
    print(json.dumps(out,indent=2,default=str));eng.dispose()
if __name__=='__main__':main()
