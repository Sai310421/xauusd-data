from __future__ import annotations
import argparse,json,math,itertools
from collections import deque,Counter
from pathlib import Path
from datetime import datetime,timezone,timedelta
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

BAL0=1000.;LEV=2000.;CS=100.;MAXLOT=3.;MAXPOS=12;MAXSPR=.80;DLY=3.5;HDD=20.;MINML=600.;STOPL=2
TP=.8575;ADD=3.11;EMG=5.75;MAXLAY=9;FM=1.4666666667;LM=1.5
DDR_CAP=18.;FPW=400;FPL=.60;FPH=.80;MINS=.50;CB=6.

class BT:
 def __init__(s,base,rs):
  s.BASE=base;s.RS=rs;s.b=BAL0;s.pk=BAL0;s.dk=None;s.ds=BAL0;s.ts=deque(maxlen=512);s.dsamp=deque(maxlen=512);s.pm=None;s.tc=0;s.psig=0
  s.side=0;s.legs=[];s.op=None;s.lc=None;s.res=False;s.pkb=0;s.tr=[];s.rej=Counter();s.ex=Counter();s.cl=0;s.mxlay=0;s.mxlot=0;s.minml=1e9;s.mxdd=0;s.con=0;s.fp=[];s.ddrs=[]
 def lf(s,x):return max(.01,math.floor((x+1e-12)/.01)*.01)
 def ll(s,n):return s.lf(s.BASE if n<=0 else s.BASE*FM if n==1 else s.BASE*(LM**n))
 def lots(s):return sum(x[1] for x in s.legs)
 def be(s):q=s.lots();return sum(x[0]*x[1] for x in s.legs)/q if q else 0
 def pnl(s,bid,ask):
  m=bid if s.side>0 else ask;return sum((m-x[0])*s.side*x[1]*CS for x in s.legs)
 def eq(s,bid,ask):return s.b+s.pnl(bid,ask)
 def ml(s,bid,ask):
  q=s.lots()
  if q<=0:return 99999.
  m=((bid+ask)/2)*CS*q/LEV;return 100*s.eq(bid,ask)/m if m>0 else 99999.
 def dd(s,bid,ask):
  e=s.eq(bid,ask);s.pk=max(s.pk,e);d=100*max(0,s.pk-e)/s.pk if s.pk else 0;s.mxdd=max(s.mxdd,d);return d
 def day(s,ns):return (datetime.fromtimestamp(ns/1e9,tz=timezone.utc)+timedelta(hours=9)).date()
 def resetd(s,ns):
  k=s.day(ns)
  if k!=s.dk:s.dk=k;s.ds=s.b;s.con=0
 def dly(s,bid,ask):return 100*max(0,s.ds-s.eq(bid,ask))/s.ds if s.ds else 0
 def risk(s,lot,bid,ask,ns):
  s.resetd(ns)
  if ask-bid>MAXSPR:s.rej['spread']+=1;return False
  if s.dly(bid,ask)>=DLY:s.rej['daily_dd']+=1;return False
  if s.dd(bid,ask)>=HDD:s.rej['fdd']+=1;return False
  if s.ml(bid,ask)<MINML:s.rej['margin']+=1;return False
  if s.con>=STOPL:s.rej['loss_streak']+=1;return False
  j=datetime.fromtimestamp(ns/1e9,tz=timezone.utc)+timedelta(hours=9)
  if j.weekday()==4 and j.hour>=21:s.rej['friday']+=1;return False
  if len(s.legs)>=MAXPOS:s.rej['maxpos']+=1;return False
  if s.lots()+lot>MAXLOT+1e-12:s.rej['maxlot']+=1;return False
  return True
 def sig(s,m):
  if s.pm is not None:s.dsamp.append(m-s.pm)
  s.pm=m;s.tc+=1
  if s.tc<5:return 0
  s.tc=0;s.ts.append(m)
  if len(s.ts)<13:return 0
  a=list(s.ts);ma=lambda n:sum(a[-n:])/n
  f,sl,c1,c2=ma(3),ma(5),ma(8),ma(13)
  r=1 if m>f>sl>c1>c2 else -1 if m<f<sl<c1<c2 else 0
  fire=r if r and r!=s.psig else 0
  if r:s.psig=r
  return fire
 def ddr(s,bid,ask):
  d=s.dd(bid,ask);x=min(1,max(0,DDR_CAP-d)/DDR_CAP);den=1-math.exp(-2.5);r=(1-math.exp(-2.5*x))/den
  sc=min(1,max(.5,r));s.ddrs.append(sc);return sc
 def fpp(s,bid,ask):
  n=min(len(s.dsamp),FPW)
  if n<40:return .5
  xs=[x*s.side for x in list(s.dsamp)[-n:]];mu=sum(xs)/n;v=sum((x-mu)**2 for x in xs)/max(n-1,1);sg=math.sqrt(max(v,1e-12))
  m=bid if s.side>0 else ask;adv=abs(m-s.legs[-1][0]);z=(EMG-adv-max(mu,0)*n)/(sg*math.sqrt(n)+1e-12);p=1/(1+math.exp(max(-60,min(60,1.702*z))))
  s.fp.append(p);return p
 def scale(s,bid,ask):
  p=s.fpp(bid,ask);f=1 if p<=FPL else MINS if p>=FPH else 1-(p-FPL)/(FPH-FPL)*(1-MINS)
  return min(1,max(MINS,s.ddr(bid,ask)*f))
 def vol(s):
  n=min(len(s.dsamp),200)
  if n<80:return 999
  x=[z*s.side for z in list(s.dsamp)[-n:]];m=sum(x)/n;return math.sqrt(sum((z-m)**2 for z in x)/max(1,n-1))
 def open(s,lot,bid,ask,ns,tag):
  lot=s.lf(lot)
  if not s.risk(lot,bid,ask,ns):return False
  p=ask if s.side>0 else bid;s.legs.append((p,lot,ns,tag));s.op=s.op or ns;s.mxlay=max(s.mxlay,len(s.legs));s.mxlot=max(s.mxlot,s.lots());s.minml=min(s.minml,s.ml(bid,ask));return True
 def close(s,bid,ask,ns,why):
  p=s.pnl(bid,ask);q=s.lots();s.b+=p;s.cl+=q;s.tr.append((p,q,len(s.legs),why));s.ex[why]+=1;s.con=s.con+1 if p<0 else 0
  s.side=0;s.legs=[];s.op=None;s.lc=ns;s.res=False;s.pkb=0
 def tick(s,t):
  bid=float(t.bid_price);ask=float(t.ask_price);ns=int(t.ts_event);m=(bid+ask)/2;s.resetd(ns);s.dd(bid,ask)
  if s.legs:
   ml=s.ml(bid,ask);s.minml=min(s.minml,ml);d=s.dd(bid,ask)
   if d>=HDD:return s.close(bid,ask,ns,'hard_fdd')
   if ml<MINML:return s.close(bid,ask,ns,'margin')
   if s.dly(bid,ask)>=DLY:return s.close(bid,ask,ns,'daily_dd')
   p=s.pnl(bid,ask);s.pkb=max(s.pkb,p)
   if s.pkb>=10 and p<=s.pkb*.6:return s.close(bid,ask,ns,'giveback')
   if s.op and ns-s.op>=180*60*1e9:return s.close(bid,ask,ns,'max_hold')
   mark=bid if s.side>0 else ask
   if (mark-s.be())*s.side>=TP:return s.close(bid,ask,ns,'basket_tp')
   last=s.legs[-1][0];adv=last-mark if s.side>0 else mark-last;n=len(s.legs)
   if n<MAXLAY and adv>=ADD:
    raw=s.ll(n);sc=s.scale(bid,ask) if n>=s.RS else 1;s.open(raw*sc,bid,ask,ns,'add');return
   if n>=MAXLAY and adv>=EMG:
    bed=(s.be()-mark)*s.side;ok=bed<=6.2 or (d<19 and s.vol()<=.12)
    if ok and not s.res:
     if s.open(s.lots()*.06,bid,ask,ns,'rescue'):s.res=True
    else:return s.close(bid,ask,ns,'emergency')
   if s.res and s.pnl(bid,ask)>=0:return s.close(bid,ask,ns,'fast_complete')
   return
  if s.lc is not None and ns-s.lc<10e9:return
  dt=datetime.fromtimestamp(ns/1e9,tz=timezone.utc)
  if not 7<=dt.hour<17:return
  x=s.sig(m)
  if x:s.side=x;s.open(s.BASE,bid,ask,ns,'seed') or setattr(s,'side',0)
 def out(s,n,a,b):
  ps=[x[0] for x in s.tr];w=[x for x in ps if x>0];l=[x for x in ps if x<0];gp=sum(w);gl=-sum(l);cb=s.cl*CB
  return {'base_lot':s.BASE,'risk_start_layer':s.RS,'start_balance':BAL0,'final_balance_ex_cb':s.b,'return_pct_ex_cb':100*(s.b/BAL0-1),'cashback_est':cb,'return_pct_inc_cb':100*((s.b+cb)/BAL0-1),'trades':len(ps),'win_rate_pct':100*len(w)/len(ps) if ps else 0,'profit_factor':gp/gl if gl else (999 if gp else 0),'max_floating_dd_pct':s.mxdd,'min_margin_level_pct':s.minml if s.minml<1e9 else None,'max_layers':s.mxlay,'max_total_lots':s.mxlot,'closed_lots':s.cl,'mean_ddr_scale':sum(s.ddrs)/len(s.ddrs) if s.ddrs else None,'mean_fp':sum(s.fp)/len(s.fp) if s.fp else None,'ddr_uses':len(s.ddrs),'fp_uses':len(s.fp),'exit_reasons':dict(s.ex),'entry_rejects':dict(s.rej),'raw_ticks':n,'days':(b-a)/86400e9}
def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);a=p.parse_args();c=ParquetDataCatalog(a.catalog);i=next(x for x in c.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');t=c.query(data_cls=QuoteTick,identifiers=[i.id.value])
 if not t:raise SystemExit('no ticks')
 rows=[]
 for base,rs in itertools.product([.05,.075,.10],[2,3,4]):
  b=BT(base,rs)
  for x in t:b.tick(x)
  if b.legs:b.close(float(t[-1].bid_price),float(t[-1].ask_price),int(t[-1].ts_event),'eod')
  rows.append(b.out(len(t),int(t[0].ts_event),int(t[-1].ts_event)))
 rows=sorted(rows,key=lambda r:(r['max_floating_dd_pct'], -r['profit_factor'], -r['return_pct_ex_cb']))
 out={'verification_level':'RAW_BIDASK_GRID_V2','rows':rows,'selection_note':'No winner auto-selected; inspect PF/return/DD tradeoff.'}
 q=Path(a.out);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
