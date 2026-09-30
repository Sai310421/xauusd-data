#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, math, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

DATA=Path("csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv")
OUT=Path("bt_results/ae_a17_ml66_pf12_v2")
OUT.mkdir(parents=True,exist_ok=True)
SPREAD_USD=.80
HORIZON=72
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

def labels(raw):
    a=atr(raw).to_numpy(); o=raw.Open.to_numpy(); h=raw.High.to_numpy(); l=raw.Low.to_numpy(); c=raw.Close.to_numpy(); n=len(raw)
    br=np.full(n,np.nan); sr=np.full(n,np.nan); be=np.full(n,-1,int); se=np.full(n,-1,int)
    half=SPREAD_USD/2
    for i in range(n-HORIZON-1):
        if not np.isfinite(a[i]) or a[i]<=0: continue
        risk=SL_ATR*a[i]; eb=o[i+1]+half; es=o[i+1]-half
        bsl=eb-risk; btp=eb+RR*risk; ssl=es+risk; stp=es-RR*risk
        last=min(n-1,i+HORIZON); bo=so=None; bx=sx=last
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

def split(n):
    a=int(n*.60); b=int(n*.80)
    return np.arange(0,max(0,a-PURGE)),np.arange(min(n,a+PURGE),max(min(n,b-PURGE),min(n,a+PURGE))),np.arange(min(n,b+PURGE),n)

def fit(X,y,tr,va,te,side):
    good=np.isfinite(y)&np.isfinite(X).all(axis=1); tr=tr[good[tr]]; va=va[good[va]]; te=te[good[te]]
    reg=LGBMRegressor(n_estimators=1200,learning_rate=.02,num_leaves=15,max_depth=5,subsample=.85,colsample_bytree=.8,reg_lambda=2.,reg_alpha=.2,random_state=42,verbosity=-1)
    reg.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],callbacks=[early_stopping(80,verbose=False)])
    pva=reg.predict(X[va]); pte=reg.predict(X[te]); ptr=reg.predict(X[tr])
    cls=LGBMClassifier(n_estimators=800,learning_rate=.02,num_leaves=15,max_depth=5,subsample=.85,colsample_bytree=.8,reg_lambda=2.,reg_alpha=.2,random_state=49,verbosity=-1)
    cls.fit(np.c_[X[tr],ptr],(y[tr]>0).astype(int),eval_set=[(np.c_[X[va],pva],(y[va]>0).astype(int))],callbacks=[early_stopping(60,verbose=False)])
    qva=cls.predict_proba(np.c_[X[va],pva])[:,1]; qte=cls.predict_proba(np.c_[X[te],pte])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(qva,(y[va]>0).astype(int))
    cva=iso.predict(qva); cte=iso.predict(qte)
    best=(.55,-1e99)
    for t in np.arange(.50,.81,.01):
        m=(cva>=t)&(pva>0)
        if m.sum()<25: continue
        yy=y[va][m]; gp=yy[yy>0].sum(); gl=-yy[yy<0].sum(); pf=gp/gl if gl>0 else 99.
        score=yy.sum()+3*math.log(max(pf,.01))-0.002*m.sum()
        if score>best[1]: best=(float(t),float(score))
    joblib.dump(reg,OUT/f"{side}_expected_R.joblib"); joblib.dump(cls,OUT/f"{side}_meta.joblib"); joblib.dump(iso,OUT/f"{side}_calibrator.joblib")
    return {"reg":reg,"cls":cls,"iso":iso,"threshold":best[0],"tr":tr,"va":va,"te":te,"pva":pva,"cva":cva,"pred":pte,"prob":cte,
      "mae":float(mean_absolute_error(y[te],pte)),
      "auc":float(roc_auc_score((y[te]>0).astype(int),qte)),
      "brier":float(brier_score_loss((y[te]>0).astype(int),cte))}

def evaluate(name,idx,bp,sp,bc,sc,bt,st,bexit,sexit,mode):
    trades=[]; k=0
    while k<len(idx):
        i=int(idx[k]); er_b=float(bp[k]); er_s=float(sp[k]); pb=float(bc[k]); ps=float(sc[k])
        side=None
        if mode=="ml":
            if max(er_b,er_s)>0: side="BUY" if er_b>=er_s else "SELL"
        else:
            ub=er_b*pb if er_b>0 and pb>=bt else -1e9
            us=er_s*ps if er_s>0 and ps>=st else -1e9
            if max(ub,us)>-1e8:
                side="BUY" if ub>=us else "SELL"
                if mode=="a17":
                    er,p=(er_b,pb) if side=="BUY" else (er_s,ps)
                    # Label already includes spread; A17 adds uncertainty/tail safety margin, not duplicated transaction cost.
                    safety=.04 + .16*(1-p)
                    if er*p<=safety: side=None
        if side is None: k+=1; continue
        r=float(BUY_R[i] if side=="BUY" else SELL_R[i]); ex=int(BUY_EXIT[i] if side=="BUY" else SELL_EXIT[i])
        if not np.isfinite(r) or ex<=i: k+=1; continue
        trades.append((i,ex,side,r))
        while k<len(idx) and int(idx[k])<=ex: k+=1
    rs=np.array([x[3] for x in trades],float)
    if len(rs)==0: return {"trades":0}
    gp=rs[rs>0].sum(); gl=-rs[rs<0].sum(); eq=np.cumsum(rs); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq
    # Approx account return assuming fixed 1% initial-equity risk/trade; descriptive, not compounding.
    return {"trades":int(len(rs)),"WR":float((rs>0).mean()),"PF":float(gp/gl) if gl>0 else None,
      "sum_R":float(rs.sum()),"mean_R":float(rs.mean()),"maxDD_R":float(dd.max() if len(dd) else 0),
      "return_pct_at_1pct_fixed_risk":float(rs.sum()*RISK_PCT),"maxDD_pct_at_1pct_fixed_risk":float((dd.max() if len(dd) else 0)*RISK_PCT),
      "first_idx":int(trades[0][0]),"last_idx":int(trades[-1][0])}


def evaluate_thresholds(idx,bp,sp,bthr,sthr,bexit,sexit):
    trades=[]; k=0
    while k<len(idx):
        i=int(idx[k]); eb=float(bp[k]); es=float(sp[k]); side=None
        ub=eb if eb>=bthr else -1e9; us=es if es>=sthr else -1e9
        if max(ub,us)>-1e8: side="BUY" if ub>=us else "SELL"
        if side is None: k+=1; continue
        r=float(BUY_R[i] if side=="BUY" else SELL_R[i]); ex=int(BUY_EXIT[i] if side=="BUY" else SELL_EXIT[i])
        if not np.isfinite(r) or ex<=i: k+=1; continue
        trades.append((i,ex,side,r))
        while k<len(idx) and int(idx[k])<=ex: k+=1
    if not trades: return {"trades":0,"PF":0.0,"sum_R":0.0,"maxDD_R":999.0}
    rs=np.array([x[3] for x in trades],float); gp=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    pf=float(gp/gl) if gl>0 else 99.0
    eq=np.cumsum(rs); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq
    return {"trades":int(len(rs)),"PF":pf,"sum_R":float(rs.sum()),"WR":float((rs>0).mean()),"mean_R":float(rs.mean()),"maxDD_R":float(dd.max() if len(dd) else 0)}

raw=load(); X,base=build66(raw)
BUY_R,SELL_R,BUY_EXIT,SELL_EXIT=labels(raw)
n=min(len(X),len(raw)); X=X[-n:]; raw=raw.iloc[-n:].reset_index(drop=True)
BUY_R=BUY_R[-n:]; SELL_R=SELL_R[-n:]; BUY_EXIT=BUY_EXIT[-n:]; SELL_EXIT=SELL_EXIT[-n:]
tr,va,te=split(n)
b=fit(X,BUY_R,tr,va,te,"BUY"); s=fit(X,SELL_R,tr,va,te,"SELL")
assert np.array_equal(b["te"],s["te"])
idx=b["te"]
# Tune only on validation: direction-specific minimum Expected-R thresholds.
grid=np.arange(0.00,0.525,0.025)
cands=[]
for tb in grid:
    for ts in grid:
        vm=evaluate_thresholds(b["va"],b["pva"],s["pva"],float(tb),float(ts),BUY_EXIT,SELL_EXIT)
        if vm["trades"]<70: continue
        # Prefer PF >= 1.20, then total R, while mildly penalizing DD.
        hit=1 if vm["PF"]>=1.20 else 0
        score=(hit, vm["PF"], vm["sum_R"]-0.10*vm["maxDD_R"], vm["trades"])
        cands.append((score,float(tb),float(ts),vm))
cands.sort(key=lambda x:x[0],reverse=True)
best_t=cands[0] if cands else ((0,0,0,0),0.0,0.0,{"trades":0,"PF":0.0})
tb,ts=best_t[1],best_t[2]
oos_tuned=evaluate_thresholds(idx,b["pred"],s["pred"],tb,ts,BUY_EXIT,SELL_EXIT)
variants={
 "ML66_only":evaluate("ML66_only",idx,b["pred"],s["pred"],b["prob"],s["prob"],b["threshold"],s["threshold"],BUY_EXIT,SELL_EXIT,"ml"),
 "ML66_ER_Tuned":oos_tuned,
 "ML66_Meta":evaluate("ML66_Meta",idx,b["pred"],s["pred"],b["prob"],s["prob"],b["threshold"],s["threshold"],BUY_EXIT,SELL_EXIT,"meta"),
 "ML66_Meta_A17":evaluate("ML66_Meta_A17",idx,b["pred"],s["pred"],b["prob"],s["prob"],b["threshold"],s["threshold"],BUY_EXIT,SELL_EXIT,"a17")
}
pred=pd.DataFrame({"idx":idx,"datetime":raw.DateTime.iloc[idx].astype(str).to_numpy(),"buy_er":b["pred"],"buy_p":b["prob"],"sell_er":s["pred"],"sell_p":s["prob"]})
pred.to_csv(OUT/"oos_predictions.csv",index=False)
report={
 "status":"OOS_RESEARCH_ONLY_NOT_LIVE_APPROVED",
 "data_file":str(DATA),"data_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
 "rows":int(len(raw)),"start":str(raw.DateTime.min()),"end":str(raw.DateTime.max()),
 "features":66,"feature_order":FEATURES,
 "label":{"spread_usd":SPREAD_USD,"horizon_bars":HORIZON,"sl_atr":SL_ATR,"rr":RR,"same_bar_rule":"SL_first"},
 "split":{"train":int(len(tr)),"validation":int(len(va)),"oos":int(len(te)),"purge":PURGE},
 "BUY_model":{"threshold":b["threshold"],"test_MAE":b["mae"],"test_AUC":b["auc"],"test_Brier":b["brier"]},
 "SELL_model":{"threshold":s["threshold"],"test_MAE":s["mae"],"test_AUC":s["auc"],"test_Brier":s["brier"]},\n "validation_er_threshold_search":{"target_PF":1.20,"buy_threshold":tb,"sell_threshold":ts,"validation_metrics":best_t[3]},
 "variants":variants,
 "notes":["All thresholds chosen before OOS using validation only.","OOS simulation is sequential/non-overlapping: next signal allowed only after prior trade exit.","Return% proxy assumes fixed 1% of initial equity risk per trade; final portfolio return requires Nautilus/raw-tick execution validation.","Spread proxy included in labels; commission/slippage/swap are not included."]
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
