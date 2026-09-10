from __future__ import annotations
import argparse, json, math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

FEATURE_NAMES = [
"RSI_feat_lag_0","ADX_feat_lag_0","MACD_Diff_ATR_Ratio_lag_0","BB_Width_ATR_Ratio_lag_0","EMA_Diff_ATR_Ratio_lag_0","Momentum_5_lag_0","Momentum_15_lag_0","FracDiff_LogPrice_lag_0","ATR_Ratio_lag_0","Spread_ATR_Ratio_lag_0","Hour_Sin_feat_lag_0","Hour_Cos_feat_lag_0","Day_Sin_feat_lag_0","Day_Cos_feat_lag_0","Hour_Activity_feat_lag_0",
"RSI_feat_lag_1","ADX_feat_lag_1","MACD_Diff_ATR_Ratio_lag_1","BB_Width_ATR_Ratio_lag_1","EMA_Diff_ATR_Ratio_lag_1","Momentum_5_lag_1","Momentum_15_lag_1","FracDiff_LogPrice_lag_1","ATR_Ratio_lag_1","Spread_ATR_Ratio_lag_1","Hour_Sin_feat_lag_1","Hour_Cos_feat_lag_1","Day_Sin_feat_lag_1","Day_Cos_feat_lag_1","Hour_Activity_feat_lag_1",
"RSI_feat_lag_3","ADX_feat_lag_3","MACD_Diff_ATR_Ratio_lag_3","BB_Width_ATR_Ratio_lag_3","EMA_Diff_ATR_Ratio_lag_3","Momentum_5_lag_3","Momentum_15_lag_3","FracDiff_LogPrice_lag_3","ATR_Ratio_lag_3","Spread_ATR_Ratio_lag_3","Hour_Sin_feat_lag_3","Hour_Cos_feat_lag_3","Day_Sin_feat_lag_3","Day_Cos_feat_lag_3","Hour_Activity_feat_lag_3",
"RSI_feat_lag_7","ADX_feat_lag_7","MACD_Diff_ATR_Ratio_lag_7","BB_Width_ATR_Ratio_lag_7","EMA_Diff_ATR_Ratio_lag_7","Momentum_5_lag_7","Momentum_15_lag_7","FracDiff_LogPrice_lag_7","ATR_Ratio_lag_7","Spread_ATR_Ratio_lag_7","Hour_Sin_feat_lag_7","Hour_Cos_feat_lag_7","Day_Sin_feat_lag_7","Day_Cos_feat_lag_7","Hour_Activity_feat_lag_7",
"RSI_60m","ADX_60m","EMA_Diff_ATR_Ratio_60m","RSI_240m","ADX_240m","EMA_Diff_ATR_Ratio_240m"]

@dataclass
class Bar:
    ts:int; o:float; h:float; l:float; c:float; spread:float; n:int

def ema(a, span):
    if len(a) < span: return np.nan
    alpha=2/(span+1); e=float(a[0])
    for v in a[1:]: e=alpha*float(v)+(1-alpha)*e
    return e

def atr(bars, n=14):
    if len(bars)<n+1:return np.nan
    tr=[]
    for i in range(-n,0):
        b=bars[i]; pc=bars[i-1].c
        tr.append(max(b.h-b.l,abs(b.h-pc),abs(b.l-pc)))
    return float(np.mean(tr))

def rsi(bars,n=14):
    if len(bars)<n+1:return np.nan
    x=np.array([b.c for b in bars[-(n+1):]],float); d=np.diff(x); up=np.maximum(d,0).mean(); dn=np.maximum(-d,0).mean()
    return 100.0 if dn==0 else 100-(100/(1+up/dn))

def adx(bars,n=14):
    if len(bars)<2*n+1:return np.nan
    tr=[]; plus=[]; minus=[]; bs=bars[-(2*n+1):]
    for i in range(1,len(bs)):
        b,p=bs[i],bs[i-1]; tr.append(max(b.h-b.l,abs(b.h-p.c),abs(b.l-p.c)))
        u=b.h-p.h; d=p.l-b.l; plus.append(u if u>d and u>0 else 0); minus.append(d if d>u and d>0 else 0)
    tr=np.array(tr); plus=np.array(plus); minus=np.array(minus); dx=[]
    for j in range(n,len(tr)+1):
        t=tr[j-n:j].sum(); p=100*plus[j-n:j].sum()/max(t,1e-9); m=100*minus[j-n:j].sum()/max(t,1e-9); dx.append(100*abs(p-m)/max(p+m,1e-9))
    return float(np.mean(dx[-n:])) if len(dx)>=n else np.nan

def fracdiff_log(bars,d=.4,k=10):
    if len(bars)<k:return np.nan
    w=[1.0]
    for i in range(1,k):w.append(-w[-1]*(d-i+1)/i)
    x=np.log(np.array([b.c for b in bars[-k:]],float)); return float(np.dot(np.array(w[::-1]),x))

def base15(bars,ts):
    if len(bars)<55:return None
    closes=np.array([b.c for b in bars],float); a14=atr(bars,14); a50=atr(bars,50)
    if not np.isfinite(a14) or a14<=0:return None
    e9=ema(closes[-30:],9);e21=ema(closes[-40:],21);e12=ema(closes[-40:],12);e26=ema(closes[-50:],26)
    macd=e12-e26; macd_series=[]
    if len(closes)>=60:
        for j in range(-12,0):
            z=closes[:j] if j else closes
            if len(z)>=50: macd_series.append(ema(z[-40:],12)-ema(z[-50:],26))
    sig=ema(np.array(macd_series,float),9) if len(macd_series)>=9 else 0.0
    std20=float(np.std(closes[-20:],ddof=0)); hour=(ts//3600)%24; day=(ts//86400)%7
    recent_n=np.array([b.n for b in bars[-48:]],float); act=bars[-1].n/max(float(np.median(recent_n)),1.0)
    return [rsi(bars),adx(bars),(macd-sig)/a14,(4*std20)/a14,(e9-e21)/a14,(closes[-1]-closes[-6])/a14,(closes[-1]-closes[-16])/a14,fracdiff_log(bars),a14/max(a50,1e-9),bars[-1].spread/a14,math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24),math.sin(2*math.pi*day/7),math.cos(2*math.pi*day/7),act]

def htf3(bars):
    if len(bars)<55:return [np.nan]*3
    a=atr(bars,14);cl=np.array([b.c for b in bars],float)
    return [rsi(bars),adx(bars),(ema(cl[-30:],9)-ema(cl[-40:],21))/max(a,1e-9)]

class FeatureEngine:
    def __init__(self):
        self.buckets={300:None,3600:None,14400:None}; self.cur={}; self.bars={300:deque(maxlen=600),3600:deque(maxlen=300),14400:deque(maxlen=200)}; self.m5_base=deque(maxlen=32); self.latest=None
    def update(self,ts,mid,spread):
        for tf in (300,3600,14400):
            b=ts//tf
            if self.buckets[tf] is None:self.buckets[tf]=b;self.cur[tf]=[ts,mid,mid,mid,mid,spread,1]
            elif b!=self.buckets[tf]:
                x=self.cur[tf];bar=Bar(x[0],x[1],x[2],x[3],x[4],x[5]/x[6],x[6]);self.bars[tf].append(bar);self.buckets[tf]=b;self.cur[tf]=[ts,mid,mid,mid,mid,spread,1]
                if tf==300:
                    f=base15(list(self.bars[300]),bar.ts)
                    if f is not None:self.m5_base.append(f)
                    self._refresh()
            else:
                x=self.cur[tf];x[2]=max(x[2],mid);x[3]=min(x[3],mid);x[4]=mid;x[5]+=spread;x[6]+=1
    def _refresh(self):
        if len(self.m5_base)<8:return
        v=[]
        for lag in (0,1,3,7):v.extend(self.m5_base[-1-lag])
        v.extend(htf3(list(self.bars[3600])));v.extend(htf3(list(self.bars[14400])))
        arr=np.array(v,dtype=np.float32)
        if len(arr)==66 and np.all(np.isfinite(arr)):self.latest=arr
    def regime(self):
        if self.latest is None:return 'UNKNOWN'
        adx5=self.latest[1]; bb=self.latest[3]; atrr=self.latest[8]
        if atrr>1.35 or bb>4.5:return 'FLASH_VOL'
        if adx5>=28:return 'TREND_BREAK'
        if adx5<=18 and bb<2.2:return 'RANGE_GRID'
        return 'SWING_TRANSITION'

class Recorder:
    def __init__(self,gate='C2',trigger=.12,add=.025,reversal=.20,max_layers=10,confirm1=.025,confirm2=.05,pullback=.04):
        self.gate=gate;self.trigger=trigger;self.add=add;self.reversal=reversal;self.max_layers=max_layers;self.confirm1=confirm1;self.confirm2=confirm2;self.pullback=pullback
        self.bucket=None;self.anchor=None;self.started=False;self.pending=0;self.trigger_px=None;self.pending_ext=None;self.mids=deque(maxlen=16);self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None;self.trade=None;self.records=[]
    def need(self):return self.confirm1 if self.gate=='C1' else self.confirm2
    def close(self,ts,bid,ask,reason):
        if not self.active:return
        px=bid if self.side>0 else ask;p=sum((px-e)*self.side for e in self.entries)
        r=self.trade.copy();r.update(exit_ts=ts,pnl=p,win=int(p>0),reason=reason,layers=len(self.entries));self.records.append(r)
        self.active=False;self.side=0;self.entries=[];self.trade=None
    def update(self,ts,bid,ask,feat,regime):
        mid=(bid+ask)/2;self.mids.append(mid);b=ts//60
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            if self.active:
                px=bid if self.side>0 else ask;rev=px<=self.extreme-self.reversal if self.side>0 else px>=self.extreme+self.reversal
                if rev:self.close(ts,bid,ask,'REV')
            self.bucket=b;self.anchor=self.mids[-2] if len(self.mids)>1 else mid;self.started=False;self.pending=0
        if not self.active and not self.started:
            if self.pending==0:
                up=mid>=self.anchor+self.trigger;dn=mid<=self.anchor-self.trigger
                if up or dn:self.pending=1 if up else -1;self.trigger_px=mid;self.pending_ext=mid
            else:
                s=self.pending;self.pending_ext=max(self.pending_ext,mid) if s>0 else min(self.pending_ext,mid);ext=(self.pending_ext-self.trigger_px)*s;pb=(self.pending_ext-mid)*s
                if pb>self.pullback:self.started=True;self.pending=0
                elif ext>=self.need() and (self.gate!='C3' or (len(self.mids)>=6 and (self.mids[-1]-self.mids[-6])*s>0)):
                    if feat is None:self.started=True;self.pending=0
                    else:
                        entry=ask if s>0 else bid;self.side=s;self.active=True;self.entries=[entry];self.last_add=entry;self.extreme=bid if s>0 else ask;self.started=True;self.pending=0;self.trade={'entry_ts':ts,'side':s,'features':feat.copy(),'regime':regime}
        if self.active:
            px=bid if self.side>0 else ask;self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
            while len(self.entries)<self.max_layers:
                tgt=self.last_add+self.side*self.add;cross=px>=tgt if self.side>0 else px<=tgt
                if not cross:break
                fill=ask if self.side>0 else bid;self.entries.append(fill);self.last_add=tgt
    def finish(self,ts,bid,ask):
        if self.active:self.close(ts,bid,ask,'EOD')

def pf(pnls):
    p=np.asarray(pnls,float);gp=p[p>0].sum();gl=-p[p<0].sum();return float(gp/gl) if gl>0 else (float('inf') if gp>0 else 0.0)

def maxdd_from_pnls(pnls,initial=1000.):
    eq=initial+np.cumsum(np.asarray(pnls,float)); peak=np.maximum.accumulate(np.r_[initial,eq]); vals=np.r_[initial,eq]; return float(np.max((peak-vals)/np.maximum(peak,1e-9))*100)

def choose_threshold(y,p,pnl,coverage_min=.30):
    best=(.55,-1e99)
    for t in np.linspace(.35,.80,46):
        keep=p<t
        if keep.mean()<coverage_min or keep.sum()<20:continue
        score=pf(pnl[keep]) + .002*keep.sum() - .5*max(0,0.5-keep.mean())
        if score>best[1]:best=(float(t),float(score))
    return best[0]

def summarize(df,name):
    p=df.pnl.to_numpy(float);n=len(df);return {'name':name,'N':n,'WR_pct':100*float((p>0).mean()) if n else 0.,'PF':pf(p),'return_pct_on_1000':float(p.sum()/10),'max_DD_pct':maxdd_from_pnls(p),'gross_usd':float(p.sum())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--gate',choices=['C1','C2','C3'],default='C2');ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    fe=FeatureEngine();rec=Recorder(a.gate);last=None
    for t in ticks:
        bid=float(t.bid_price.as_double());ask=float(t.ask_price.as_double());ts=int(t.ts_event)//1_000_000_000;mid=(bid+ask)/2;fe.update(ts,mid,ask-bid);rec.update(ts,bid,ask,fe.latest,fe.regime());last=(ts,bid,ask)
    if last:rec.finish(*last)
    rows=[]
    for r in rec.records:
        d={k:v for k,v in r.items() if k!='features'}
        for i,n in enumerate(FEATURE_NAMES):d[n]=float(r['features'][i])
        rows.append(d)
    df=pd.DataFrame(rows).sort_values('entry_ts').reset_index(drop=True)
    if len(df)<80:raise SystemExit(f'Insufficient labeled trades: {len(df)}')
    X=df[FEATURE_NAMES].to_numpy(np.float32);y=(df.pnl.to_numpy(float)<0).astype(int);pnl=df.pnl.to_numpy(float)
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,brier_score_loss
    probs=np.full(len(df),np.nan);thresholds=[];folds=[]
    starts=np.linspace(int(len(df)*.4),len(df),4,dtype=int)
    for k in range(len(starts)-1):
        train_end=starts[k];test_end=starts[k+1];tr=np.arange(train_end);te=np.arange(train_end,test_end)
        if len(np.unique(y[tr]))<2:continue
        cut=max(30,int(len(tr)*.8));fit_idx=tr[:cut];cal_idx=tr[cut:]
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=.35))
        model.fit(X[fit_idx],y[fit_idx]);raw_cal=model.predict_proba(X[cal_idx])[:,1] if len(cal_idx) else model.predict_proba(X[fit_idx])[:,1]
        yc=y[cal_idx] if len(cal_idx) else y[fit_idx];pc=pnl[cal_idx] if len(cal_idx) else pnl[fit_idx]
        cal=LogisticRegression(max_iter=500)
        if len(np.unique(yc))>=2:cal.fit(raw_cal.reshape(-1,1),yc);calp=cal.predict_proba(raw_cal.reshape(-1,1))[:,1]
        else:cal=None;calp=raw_cal
        th=choose_threshold(yc,calp,pc);raw_te=model.predict_proba(X[te])[:,1];pte=cal.predict_proba(raw_te.reshape(-1,1))[:,1] if cal is not None else raw_te;probs[te]=pte;thresholds.append(th);folds.append({'fold':k,'train_n':len(tr),'test_n':len(te),'threshold':th,'auc':float(roc_auc_score(y[te],pte)) if len(np.unique(y[te]))>1 else None,'brier':float(brier_score_loss(y[te],pte))})
    oos=df[np.isfinite(probs)].copy();oos['p_loss']=probs[np.isfinite(probs)]
    thvec=[]
    for f,th in zip(folds,thresholds):thvec.extend([th]*f['test_n'])
    oos['threshold']=np.array(thvec[:len(oos)])
    global_loss=float(y[:int(len(df)*.4)].mean());train_reg=df.iloc[:int(len(df)*.4)].groupby('regime')['win'].agg(['count','mean'])
    loss_rate={idx:1-row['mean'] for idx,row in train_reg.iterrows() if row['count']>=5}
    def reg_adj(row):
        lr=loss_rate.get(row.regime,global_loss);return float(np.clip(row.threshold-(lr-global_loss)*.18,.35,.80))
    oos['regime_threshold']=oos.apply(reg_adj,axis=1)
    nm=oos[oos.p_loss<oos.threshold].copy();nmr=oos[oos.p_loss<oos.regime_threshold].copy()
    results=[summarize(oos,'A_OOS_BASE'),summarize(nm,'B_NEG_MEMORY'),summarize(nmr,'C_NEG_MEMORY_REGIME')]
    base_wins=int((oos.pnl>0).sum());base_losses=int((oos.pnl<0).sum());keep_wins=int((nm.pnl>0).sum());keep_losses=int((nm.pnl<0).sum())
    diagnostics={'feature_count':66,'feature_order_match':list(oos[FEATURE_NAMES].columns)==FEATURE_NAMES,'walk_forward_folds':folds,'oos_n':len(oos),'loss_avoidance_pct':100*(1-keep_losses/max(base_losses,1)),'win_retention_pct':100*keep_wins/max(base_wins,1),'reject_rate_pct':100*(1-len(nm)/max(len(oos),1)),'regime_loss_rates':{k:float(v) for k,v in loss_rate.items()}}
    final_model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=.35));final_model.fit(X,y)
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'v5_labeled_trades.csv',index=False);oos.to_csv(out/'v5_oos_scored.csv',index=False)
    try:
        from skl2onnx import to_onnx
        onx=to_onnx(final_model,X[:1].astype(np.float32),target_opset=17);(out/'g75_negative_memory_v5.onnx').write_bytes(onx.SerializeToString());diagnostics['onnx_exported']=True
    except Exception as e:diagnostics['onnx_exported']=False;diagnostics['onnx_error']=str(e)
    (out/'v5_summary.json').write_text(json.dumps({'gate':a.gate,'raw_ticks':len(ticks),'period_start':man.get('start'),'period_days':man.get('days'),'chronology':'RAW_BIDASK_CAUSAL_FEATURES_WALK_FORWARD','results':results,'diagnostics':diagnostics},indent=2))
    print(json.dumps({'results':results,'diagnostics':diagnostics}))
if __name__=='__main__':main()
