#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, math, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

DATA=Path("csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv")
OUT=Path("bt_results/ae_a17_ml66_pf12_v3")
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
    br=np.full(n,np.nan); sr=np.full(n,np.nan); be=np.full(n,-1,int); se=np.full(n,-1,int)
    half=SPREAD_USD/2
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

def split4(n,purge):
    a=int(n*.50); b=int(n*.65); c=int(n*.80)
    tr=np.arange(0,max(0,a-purge))
    va=np.arange(min(n,a+purge),max(min(n,b-purge),min(n,a+purge)))
    vb=np.arange(min(n,b+purge),max(min(n,c-purge),min(n,b+purge)))
    te=np.arange(min(n,c+purge),n)
    return tr,va,vb,te

def fit_reg(X,y,tr,va,cols,seed):
    good=np.isfinite(y)&np.isfinite(X[:,cols]).all(axis=1)
    tr=tr[good[tr]]; va=va[good[va]]
    m=LGBMRegressor(n_estimators=900,learning_rate=.025,num_leaves=15,max_depth=5,
        subsample=.85,colsample_bytree=.85,reg_lambda=2.0,reg_alpha=.2,random_state=seed,verbosity=-1)
    m.fit(X[tr][:,cols],y[tr],eval_set=[(X[va][:,cols],y[va])],callbacks=[early_stopping(60,verbose=False)])
    return m

def sim(idx,bp,sp,br,sr,be,se,tb,ts):
    trades=[]; k=0
    while k<len(idx):
        i=int(idx[k]); ebp=float(bp[k]); esp=float(sp[k])
        ub=ebp if ebp>=tb else -1e9; us=esp if esp>=ts else -1e9
        if max(ub,us)<=-1e8: k+=1; continue
        side="BUY" if ub>=us else "SELL"
        r=float(br[i] if side=="BUY" else sr[i]); ex=int(be[i] if side=="BUY" else se[i])
        if not np.isfinite(r) or ex<=i: k+=1; continue
        trades.append((i,ex,side,r))
        while k<len(idx) and int(idx[k])<=ex: k+=1
    if not trades: return {"trades":0,"PF":0.0,"sum_R":0.0,"WR":0.0,"mean_R":0.0,"maxDD_R":999.0}
    rs=np.array([x[3] for x in trades],float); gp=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    pf=float(gp/gl) if gl>0 else 99.0
    eq=np.cumsum(rs); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq
    return {"trades":int(len(rs)),"PF":pf,"sum_R":float(rs.sum()),"WR":float((rs>0).mean()),
      "mean_R":float(rs.mean()),"maxDD_R":float(dd.max() if len(dd) else 0)}

raw=load(); X,base=build66(raw)
n=min(len(X),len(raw)); X=X[-n:]; raw=raw.iloc[-n:].reset_index(drop=True)
all_results=[]; survivors=[]
for horizon in HORIZONS:
    br,sr,be,se=labels_h(raw,horizon)
    tr,va,vb,te=split4(n,horizon)
    mb66=fit_reg(X,br,tr,va,np.arange(66),42+horizon)
    ms66=fit_reg(X,sr,tr,va,np.arange(66),84+horizon)
    imp=np.asarray(mb66.feature_importances_,float)+np.asarray(ms66.feature_importances_,float)
    order=np.argsort(-imp)
    for kfeat in [66,40,25,15]:
        cols=np.arange(66) if kfeat==66 else np.sort(order[:kfeat])
        if kfeat==66:
            mb,ms=mb66,ms66
        else:
            mb=fit_reg(X,br,tr,va,cols,42+horizon+kfeat)
            ms=fit_reg(X,sr,tr,va,cols,84+horizon+kfeat)
        pva_b=mb.predict(X[va][:,cols]); pva_s=ms.predict(X[va][:,cols])
        pvb_b=mb.predict(X[vb][:,cols]); pvb_s=ms.predict(X[vb][:,cols])
        pte_b=mb.predict(X[te][:,cols]); pte_s=ms.predict(X[te][:,cols])
        grid=np.arange(-0.05,0.425,0.025)
        local=[]
        for tb in grid:
            for ts in grid:
                ma=sim(va,pva_b,pva_s,br,sr,be,se,float(tb),float(ts))
                if ma["trades"]<55 or ma["PF"]<1.05 or ma["sum_R"]<=0: continue
                mbb=sim(vb,pvb_b,pvb_s,br,sr,be,se,float(tb),float(ts))
                if mbb["trades"]<45: continue
                stable_pf=min(ma["PF"],mbb["PF"])
                stable_dd=max(ma["maxDD_R"],mbb["maxDD_R"])
                score=(1 if stable_pf>=1.20 else 0, stable_pf, ma["sum_R"]+mbb["sum_R"]-.12*stable_dd)
                local.append((score,float(tb),float(ts),ma,mbb))
        local.sort(key=lambda z:z[0],reverse=True)
        if local:
            z=local[0]
            candidate={"horizon":horizon,"features":kfeat,"buy_thr":z[1],"sell_thr":z[2],
              "valA":z[3],"valB":z[4],"stable_pf":float(min(z[3]["PF"],z[4]["PF"])),
              "feature_names":[FEATURES[i] for i in cols.tolist()]}
            candidate["oos"]=sim(te,pte_b,pte_s,br,sr,be,se,z[1],z[2])
            all_results.append(candidate)
            if candidate["stable_pf"]>=1.20: survivors.append(candidate)
            joblib.dump(mb,OUT/f"H{horizon}_F{kfeat}_BUY.joblib")
            joblib.dump(ms,OUT/f"H{horizon}_F{kfeat}_SELL.joblib")

pool=survivors if survivors else all_results
pool_sorted=sorted(pool,key=lambda q:(q["stable_pf"],q["valA"]["sum_R"]+q["valB"]["sum_R"],
    -max(q["valA"]["maxDD_R"],q["valB"]["maxDD_R"])),reverse=True)
selected=pool_sorted[0] if pool_sorted else None
report={
 "status":"OOS_RESEARCH_ONLY_NOT_LIVE_APPROVED","target_pf":1.20,
 "selection_rule":"Train50 -> ValA15 search -> ValB15 stability -> OOS20; selection ignores OOS",
 "data_file":str(DATA),"data_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
 "rows":int(n),"start":str(raw.DateTime.min()),"end":str(raw.DateTime.max()),
 "horizons":HORIZONS,"feature_sizes":[66,40,25,15],"survivor_count":len(survivors),
 "selected":selected,"all_candidates":all_results,
 "notes":["PF>=1.20 must hold on BOTH ValA and ValB to be a survivor.",
 "OOS is reported but never used to choose selected candidate.",
 "Sequential non-overlapping trades; same-bar SL/TP uses SL-first.",
 "Spread proxy 0.80 USD/oz included; commission/slippage/swap not yet included.",
 "Next gate is Nautilus/raw-tick execution validation."]
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
