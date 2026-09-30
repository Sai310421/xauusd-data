#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, math, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

DATA=Path("csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv")
OUT=Path("bt_results/ae_a17_ml66_wf_v6_80pct_oos")
OUT.mkdir(parents=True,exist_ok=True)
SPREAD_USD=.80
HORIZON=72
HORIZONS=[24,48,72]
SL_ATR=1.5
RR=2.0
PURGE=72
RISK_PCT=1.0
FEATURES=[
"RSI_feat_lag_0","RSI_feat_lag_1","RSI_feat_lag_3","RSI_feat_lag_7",
"Stoch_feat_lag_0","Stoch_feat_lag_1","Stoch_feat_lag_3","Stoch_feat_lag_7",
"CCI_feat_lag_0","CCI_feat_lag_1","CCI_feat_lag_3","CCI_feat_lag_7",
"WPR_feat_lag_0","WPR_feat_lag_1","WPR_feat_lag_3","WPR_feat_lag_7",
"DeMarker_feat_lag_0","DeMarker_feat_lag_1","DeMarker_feat_lag_3","DeMarker_feat_lag_7",
"MACD_Diff_ATR_Ratio_lag_0","MACD_Diff_ATR_Ratio_lag_1","MACD_Diff_ATR_Ratio_lag_3","MACD_Diff_ATR_Ratio_lag_7",
"ADX_feat_lag_0","ADX_feat_lag_1","ADX_feat_lag_3","ADX_feat_lag_7",
"BB_Width_ATR_Ratio_lag_0","BB_Width_ATR_Ratio_lag_1","BB_Width_ATR_Ratio_lag_3","BB_Width_ATR_Ratio_lag_7",
"EMA_Diff_ATR_Ratio_lag_0","EMA_Diff_ATR_Ratio_lag_1","EMA_Diff_ATR_Ratio_lag_3","EMA_Diff_ATR_Ratio_lag_7",
"Ichimoku_Kijun_Diff_lag_0","Ichimoku_Kijun_Diff_lag_1","Ichimoku_Kijun_Diff_lag_3","Ichimoku_Kijun_Diff_lag_7",
"SAR_Diff_ATR_Ratio_lag_0","SAR_Diff_ATR_Ratio_lag_1","SAR_Diff_ATR_Ratio_lag_3","SAR_Diff_ATR_Ratio_lag_7",
"StdDev_feat_lag_0","StdDev_feat_lag_1","StdDev_feat_lag_3","StdDev_feat_lag_7",
"Momentum_5_lag_0","Momentum_5_lag_1","Momentum_5_lag_3","Momentum_5_lag_7",
"Momentum_15_lag_0","Momentum_15_lag_1","Momentum_15_lag_3","Momentum_15_lag_7",
"Sub1_RSI","Sub1_ADX","Sub1_EMA_Diff","Sub1_EMA_Slope","Sub1_Stoch",
"ATR_Ratio","Spread_ATR_Ratio","Hour_Seasonality","Day_Seasonality","Hour_Activity_feat"]

def load():
    d=pd.read_csv(DATA); d.columns=[str(x).lower() for x in d.columns]
    need=["datetime","open","high","low","close","volume"]
    miss=[x for x in need if x not in d.columns]
    if miss: raise RuntimeError(f"missing columns {miss}; got {list(d.columns)}")
    d=d[need].copy(); d["datetime"]=pd.to_datetime(d["datetime"],utc=True)
    d=d.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return pd.DataFrame({"DateTime":d.datetime,"Open":d.open.astype(float),"High":d.high.astype(float),
      "Low":d.low.astype(float),"Close":d.close.astype(float),"Volume":d.volume.astype(float),
      "Spread":SPREAD_USD})

def atr(df,p=14):
    tr=pd.concat([df.High-df.Low,(df.High-df.Close.shift()).abs(),(df.Low-df.Close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def feat(df,k):
    a=atr(df); s=df.Close; eps=1e-10
    if k in ("RSI_feat","RSI"):
        z=s.diff(); up=z.where(z>0,0).rolling(14).mean(); dn=(-z.where(z<0,0)).rolling(14).mean()
        return (100-100/(1+up/(dn+eps)))/100
    if k in ("Stoch_feat","Stoch"):
        lo=df.Low.rolling(14).min(); hi=df.High.rolling(14).max(); return (s-lo)/(hi-lo+eps)
    if k=="CCI_feat":
        tp=(df.High+df.Low+s)/3; ma=tp.rolling(20).mean()
        mad=tp.rolling(20).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        return (tp-ma)/(0.015*mad+eps)/100
    if k=="WPR_feat":
        hi=df.High.rolling(14).max(); lo=df.Low.rolling(14).min(); return (hi-s)/(hi-lo+eps)*-1
    if k=="DeMarker_feat":
        hd=df.High.diff(); ld=-df.Low.diff()
        mx=hd.where(hd>0,0).rolling(14).mean(); mn=ld.where(ld>0,0).rolling(14).mean()
        return mx/(mx+mn+eps)
    if k=="MACD_Diff_ATR_Ratio":
        m=s.ewm(span=12,adjust=False).mean()-s.ewm(span=26,adjust=False).mean()
        return (m-m.ewm(span=9,adjust=False).mean())/(a+eps)
    if k in ("ADX_feat","ADX"):
        up=df.High.diff(); dn=-df.Low.diff()
        pdm=up.where((up>dn)&(up>0),0).rolling(14).mean()
        mdm=dn.where((dn>up)&(dn>0),0).rolling(14).mean()
        tr=pd.concat([df.High-df.Low,(df.High-s.shift()).abs(),(df.Low-s.shift()).abs()],axis=1).max(axis=1).rolling(14).mean()
        pdi=100*pdm/(tr+eps); mdi=100*mdm/(tr+eps); dx=100*(pdi-mdi).abs()/(pdi+mdi+eps)
        return dx.rolling(14).mean()/100
    if k=="BB_Width_ATR_Ratio": return 4*s.rolling(20).std()/(a+eps)
    if k in ("EMA_Diff_ATR_Ratio","EMA_Diff"): return (s-s.ewm(span=200,adjust=False).mean())/(a+eps)
    if k=="Ichimoku_Kijun_Diff":
        kij=(df.High.rolling(26).max()+df.Low.rolling(26).min())/2; return (s-kij)/(a+eps)
    if k=="SAR_Diff_ATR_Ratio": return (s.ewm(span=5,adjust=False).mean()-s.ewm(span=20,adjust=False).mean())/(a+eps)
    if k=="StdDev_feat": return s.rolling(20).std()/(s+eps)
    if k=="Momentum_5": return (s-s.shift(5))/(a+eps)
    if k=="Momentum_15": return (s-s.shift(15))/(a+eps)
    if k=="EMA_Slope":
        e=s.ewm(span=200,adjust=False).mean(); return (e-e.shift(5))/(a+eps)
    if k=="ATR_Ratio": return a/(s+eps)
    if k=="Spread_ATR_Ratio": return (df.Spread)/(a+eps)
    if k=="Hour_Seasonality": return np.sin(2*np.pi*df.DateTime.dt.hour/24)
    if k=="Day_Seasonality": return np.sin(2*np.pi*df.DateTime.dt.dayofweek/7)
    if k=="Hour_Activity_feat": return pd.Series(np.where((df.DateTime.dt.hour>=8)&(df.DateTime.dt.hour<=22),1.0,.2),index=df.index)
    raise KeyError(k)

def resample(df,tf,spread=False):
    agg={"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    if spread: agg["Spread"]="mean"
    return df.set_index("DateTime").resample(tf,label="right",closed="left").agg(agg).dropna(subset=["Open","High","Low","Close"]).reset_index()

def build66(raw):
    base=resample(raw,"5min",True); h1=resample(raw,"60min",False); vals={}; cache={}
    for name in FEATURES:
        if name in {"ATR_Ratio","Spread_ATR_Ratio","Hour_Seasonality","Day_Seasonality","Hour_Activity_feat"}:
            vals[name]=feat(base,name).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(); continue
        if name.startswith("Sub1_"):
            k=name[5:]; v=feat(h1,k).replace([np.inf,-np.inf],np.nan).fillna(0)
            t=pd.DataFrame({"DateTime":h1.DateTime,name:v})
            vals[name]=pd.merge_asof(base[["DateTime"]],t,on="DateTime",direction="backward")[name].fillna(0).to_numpy(); continue
        k=name; lag=0
        for z in (0,1,3,7):
            suf=f"_lag_{z}"
            if name.endswith(suf): k=name[:-len(suf)]; lag=z; break
        if k not in cache: cache[k]=feat(base,k).replace([np.inf,-np.inf],np.nan).fillna(0)
        vals[name]=cache[k].shift(lag).fillna(0).to_numpy()
    X=np.column_stack([vals[n] for n in FEATURES]).astype(np.float32)
    assert X.shape[1]==66
    return X,base



def labels_h(raw,horizon):
    a=atr(raw).to_numpy(); o=raw.Open.to_numpy(); h=raw.High.to_numpy(); l=raw.Low.to_numpy(); c=raw.Close.to_numpy(); n=len(raw)
    br=np.full(n,np.nan); sr=np.full(n,np.nan); be=np.full(n,-1,int); se=np.full(n,-1,int); half=SPREAD_USD/2
    for i in range(n-horizon-1):
        if not np.isfinite(a[i]) or a[i]<=0: continue
        risk=SL_ATR*a[i]; eb=o[i+1]+half; es=o[i+1]-half
        bsl=eb-risk; btp=eb+RR*risk; ssl=es+risk; stp=es-RR*risk
        last=min(n-1,i+horizon); bo=so=None; bx=sx=last
        for j in range(i+1,last+1):
            if bo is None:
                if l[j]<=bsl: bo=-1.; bx=j
                elif h[j]>=btp: bo=RR; bx=j
            if so is None:
                if h[j]>=ssl: so=-1.; sx=j
                elif l[j]<=stp: so=RR; sx=j
            if bo is not None and so is not None: break
        if bo is None: bo=((c[last]-half)-eb)/risk
        if so is None: so=(es-(c[last]+half))/risk
        br[i]=bo; sr[i]=so; be[i]=bx; se[i]=sx
    return br,sr,be,se

def fit_reg(X,y,tr,va,cols,seed):
    good=np.isfinite(y)&np.isfinite(X[:,cols]).all(axis=1)
    tr=tr[good[tr]]; va=va[good[va]]
    m=LGBMRegressor(n_estimators=700,learning_rate=.03,num_leaves=15,max_depth=5,
      subsample=.85,colsample_bytree=.85,reg_lambda=2.0,reg_alpha=.2,random_state=seed,verbosity=-1)
    m.fit(X[tr][:,cols],y[tr],eval_set=[(X[va][:,cols],y[va])],callbacks=[early_stopping(50,verbose=False)])
    return m

def sim_const(idx,bp,sp,br,sr,be,se,tb,ts):
    trades=[]; k=0
    while k<len(idx):
        i=int(idx[k]); ub=float(bp[k]) if bp[k]>=tb else -1e9; us=float(sp[k]) if sp[k]>=ts else -1e9
        if max(ub,us)<=-1e8: k+=1; continue
        side="BUY" if ub>=us else "SELL"; r=float(br[i] if side=="BUY" else sr[i]); ex=int(be[i] if side=="BUY" else se[i])
        if not np.isfinite(r) or ex<=i: k+=1; continue
        trades.append((i,ex,side,r))
        while k<len(idx) and int(idx[k])<=ex: k+=1
    return metrics(trades)

def side_only(idx,bp,sp,br,sr,be,se,tb,ts,side):
    trades=[]; k=0
    while k<len(idx):
        i=int(idx[k])
        pred=float(bp[k]) if side=="BUY" else float(sp[k]); thr=tb if side=="BUY" else ts
        if pred<thr: k+=1; continue
        r=float(br[i] if side=="BUY" else sr[i]); ex=int(be[i] if side=="BUY" else se[i])
        if not np.isfinite(r) or ex<=i: k+=1; continue
        trades.append((i,ex,side,r))
        while k<len(idx) and int(idx[k])<=ex: k+=1
    return metrics(trades)

def metrics(trades):
    if not trades:return {"trades":0,"PF":0.0,"sum_R":0.0,"WR":0.0,"mean_R":0.0,"maxDD_R":999.0,"buy_trades":0,"sell_trades":0}
    rs=np.array([x[3] for x in trades],float); gp=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    eq=np.cumsum(rs); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq
    return {"trades":len(trades),"PF":float(gp/gl) if gl>0 else 99.0,"sum_R":float(rs.sum()),
      "WR":float((rs>0).mean()),"mean_R":float(rs.mean()),"maxDD_R":float(dd.max() if len(dd) else 0),
      "buy_trades":sum(1 for x in trades if x[2]=="BUY"),"sell_trades":sum(1 for x in trades if x[2]=="SELL")}

def walk_forward(X,raw,horizon,kfeat):
    br,sr,be,se=labels_h(raw,horizon); n=len(raw); start=int(n*.20); chunk=500
    decisions={}
    logs=[]
    for c0 in range(start,n-horizon,chunk):
        c1=min(n-horizon,c0+chunk); cutoff=c0-horizon
        va0=max(500,cutoff-900); tr0=max(0,va0-5000)
        tr=np.arange(tr0,max(tr0+500,va0-horizon)); va=np.arange(va0,cutoff)
        if len(tr)<1200 or len(va)<250: continue
        full=np.arange(66)
        mb0=fit_reg(X,br,tr,va,full,1000+c0+horizon)
        ms0=fit_reg(X,sr,tr,va,full,2000+c0+horizon)
        imp=np.asarray(mb0.feature_importances_,float)+np.asarray(ms0.feature_importances_,float)
        cols=full if kfeat==66 else np.sort(np.argsort(-imp)[:kfeat])
        if kfeat==66: mb,ms=mb0,ms0
        else:
            mb=fit_reg(X,br,tr,va,cols,3000+c0+horizon+kfeat)
            ms=fit_reg(X,sr,tr,va,cols,4000+c0+horizon+kfeat)
        pvb=mb.predict(X[va][:,cols]); pvs=ms.predict(X[va][:,cols])
        best=None
        for tb in np.arange(-.05,.325,.025):
            for ts in np.arange(-.05,.325,.025):
                vm=sim_const(va,pvb,pvs,br,sr,be,se,float(tb),float(ts))
                if vm["trades"]<30 or vm["sum_R"]<=0: continue
                score=(1 if vm["PF"]>=1.20 else 0,vm["PF"],vm["sum_R"]-.1*vm["maxDD_R"])
                if best is None or score>best[0]: best=(score,float(tb),float(ts),vm)
        if best is None: best=((0,0,0),0.0,0.0,{"trades":0,"PF":0.0})
        tb,ts=best[1],best[2]
        buy_vm=side_only(va,pvb,pvs,br,sr,be,se,tb,ts,"BUY")
        sell_vm=side_only(va,pvb,pvs,br,sr,be,se,tb,ts,"SELL")
        enable_buy=buy_vm["trades"]>=12 and buy_vm["PF"]>=1.20 and buy_vm["sum_R"]>0
        enable_sell=sell_vm["trades"]>=12 and sell_vm["PF"]>=1.20 and sell_vm["sum_R"]>0
        idx=np.arange(c0,c1); pb=mb.predict(X[idx][:,cols]); ps=ms.predict(X[idx][:,cols])
        for j,i in enumerate(idx):
            ub=float(pb[j]) if enable_buy and pb[j]>=tb else -1e9
            us=float(ps[j]) if enable_sell and ps[j]>=ts else -1e9
            side=None if max(ub,us)<=-1e8 else ("BUY" if ub>=us else "SELL")
            decisions[int(i)]=(side,float(pb[j]),float(ps[j]),tb,ts)
        logs.append({"chunk_start":int(c0),"chunk_end":int(c1),"train":[int(tr[0]),int(tr[-1])],
          "validation":[int(va[0]),int(va[-1])],"features":int(kfeat),"buy_thr":tb,"sell_thr":ts,
          "enable_buy":bool(enable_buy),"enable_sell":bool(enable_sell),
          "buy_validation":buy_vm,"sell_validation":sell_vm,"validation_metrics":best[3]})
    idx=np.arange(start,n-horizon); trades=[]; p=0
    while p<len(idx):
        i=int(idx[p]); d=decisions.get(i)
        if not d or d[0] is None: p+=1; continue
        side=d[0]; r=float(br[i] if side=="BUY" else sr[i]); ex=int(be[i] if side=="BUY" else se[i])
        if not np.isfinite(r) or ex<=i: p+=1; continue
        trades.append((i,ex,side,r))
        while p<len(idx) and int(idx[p])<=ex:p+=1
    return metrics(trades),logs

raw=load(); X,base=build66(raw); n=min(len(X),len(raw)); X=X[-n:]; raw=raw.iloc[-n:].reset_index(drop=True)
configs=[(72,40),(48,40)]
results=[]
for h,k in configs:
    m,logs=walk_forward(X,raw,h,k)
    results.append({"horizon":h,"features":k,"oos_walkforward":m,"chunks":logs})
selected=max(results,key=lambda q:(q["oos_walkforward"]["PF"],q["oos_walkforward"]["sum_R"],-q["oos_walkforward"]["maxDD_R"]))
report={"status":"OOS_RESEARCH_ONLY_NOT_LIVE_APPROVED","target_pf":1.20,
 "method":"strict causal walk-forward + direction gate; BUY/SELL enabled for next chunk only when prior validation side PF>=1.20 with >=12 trades",
 "data_file":str(DATA),"rows":int(n),"configs":configs,"results":results,"best_observed":selected,
 "notes":["Initial 20% is warm-up/training; remaining ~80% is causal walk-forward OOS.","Retrain chunk=500 bars; validation≈900 bars; rolling train up to 5000 bars.","Direction gate is causal and uses only preceding validation side metrics.",
 "Spread 0.80 USD/oz included; commission/slippage/swap excluded.","PF>=1.20 here is still not Raw Tick approval."]}
(OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
