from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
from statistics import mean, pstdev
import nautilus_trader
from nautilus_trader.persistence.catalog import ParquetDataCatalog

SCHEMA="AMOS.ChrisPathwayNautilusRawWFO.v1.2"
TF_MIN={"M1":1,"M5":5,"M15":15}

def _f(px):
    return float(px.as_double()) if hasattr(px,"as_double") else float(px)

def _finite(xs, minimum=1):
    out=[float(x) for x in xs]
    if len(out)<minimum or any(not math.isfinite(x) for x in out):
        raise ValueError("finite input required")
    return out

def _q(xs,q):
    s=sorted(_finite(xs)); p=(len(s)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    return s[lo] if lo==hi else s[lo]*(hi-p)+s[hi]*(p-lo)

def _cvar_losses(losses,alpha=.95):
    vals=sorted(max(0.0,float(x)) for x in losses)
    var=_q(vals,alpha); tail=[x for x in vals if x>=var]
    return mean(tail) if tail else var

def _maxdd(rs):
    wealth=1.0; peak=1.0; worst=0.0
    for r in rs:
        wealth*=max(1e-12,1+float(r)); peak=max(peak,wealth); worst=max(worst,(peak-wealth)/peak)
    return worst

def _basket(rows,weights):
    w=_finite(weights); gross=sum(abs(x) for x in w)
    if gross<=0 or len(rows)!=len(w): raise ValueError("basket dimension")
    w=[x/gross for x in w]
    return [sum(w[i]*rows[i][t] for i in range(len(rows))) for t in range(len(rows[0]))]

def _student_t(rng,df=5.0):
    z=rng.gauss(0,1); chi2=rng.gammavariate(df/2,2)
    return z/math.sqrt(max(chi2/df,1e-15))

def _regime(rs):
    r=_finite(rs,20); long_vol=pstdev(r); recent=pstdev(r[-min(20,len(r)):]); ratio=recent/max(long_vol,1e-12)
    return ("HIGH_VOL" if ratio>=1.35 else "LOW_VOL" if ratio<=.75 else "NORMAL"),ratio

def _risk_packet(rs, *, paths=120,horizon=20,seed=1,recovery=.01,tail=-.03,radius=.005):
    r=_finite(rs,20); mu=mean(r); sig=pstdev(r); regime,ratio=_regime(r); scale=min(2,max(.5,ratio))
    rng=random.Random(seed); tscale=math.sqrt(3/5); rec=bad=un=0; terminals=[]
    for _ in range(paths):
        v=0.0; hit=None
        for _step in range(horizon):
            v+=mu+sig*scale*tscale*_student_t(rng,5)
            if hit is None:
                if v>=recovery: hit="R"
                elif v<=tail: hit="T"
        terminals.append(v)
        if hit=="R": rec+=1
        elif hit=="T": bad+=1
        else: un+=1
    losses=[max(0,-x) for x in terminals]; cv=_cvar_losses(losses,.95); stressed=cv+radius
    rp=rec/paths; tp=bad/paths
    cap=max(0,min(rp,1-2*tp,1/(1+10*cv),1/(1+10*stressed)))
    return {"regime":regime,"volatility_ratio":ratio,"first_passage_recovery":rp,
            "first_passage_tail":tp,"first_passage_unresolved":un/paths,
            "cvar_loss":cv,"wasserstein_stressed_cvar":stressed,"ae_tightening_cap":cap,
            "crystal_ball_context":"RESEARCH_CONTEXT_NOT_CALIBRATED_DIRECTIONAL_PROBABILITY"}

def _rbf(x,y,l=.45): return math.exp(-.5*sum((a-b)**2 for a,b in zip(x,y))/(l*l))
def _solve(a,b):
    n=len(b); m=[list(map(float,row))+[float(b[i])] for i,row in enumerate(a)]
    for c in range(n):
        p=max(range(c,n),key=lambda r:abs(m[r][c]))
        if abs(m[p][c])<1e-12: raise ValueError("singular")
        m[c],m[p]=m[p],m[c]; d=m[c][c]; m[c]=[x/d for x in m[c]]
        for r in range(n):
            if r==c: continue
            f=m[r][c]; m[r]=[m[r][j]-f*m[c][j] for j in range(n+1)]
    return [m[i][-1] for i in range(n)]
def _dot(a,b): return sum(x*y for x,y in zip(a,b))
def _cdf(z): return .5*(1+math.erf(z/math.sqrt(2)))
def _pdf(z): return math.exp(-.5*z*z)/math.sqrt(2*math.pi)

def _optimize(rows,base,*,seed,iterations=2,candidates=8):
    base=_finite(base); gross=sum(abs(x) for x in base); base=[x/gross for x in base]; rng=random.Random(seed)
    def score(w):
        rs=_basket(rows,w); sd=pstdev(rs); raw=mean(rs)/max(sd,1e-12)
        p=_risk_packet(rs,paths=100,horizon=min(30,max(10,len(rs)//3)),seed=seed)
        return raw+.2*p["first_passage_recovery"]-.5*p["first_passage_tail"]-2*p["cvar_loss"]-2*p["wasserstein_stressed_cvar"]-(.05 if p["regime"]=="HIGH_VOL" else 0)
    xs=[base]; ys=[score(base)]
    for _ in range(min(4,len(base)+1)):
        x=[base[i]+rng.uniform(-.25,.25) for i in range(len(base))]; gross=sum(abs(v) for v in x) or 1; x=[v/gross for v in x]
        xs.append(x); ys.append(score(x))
    for _ in range(iterations):
        k=[[_rbf(xs[i],xs[j])+(1e-6 if i==j else 0) for j in range(len(xs))] for i in range(len(xs))]
        alpha=_solve(k,ys); best=max(ys); chosen=None; best_ei=-1
        for _c in range(candidates):
            x=[base[i]+rng.uniform(-.75,.75) for i in range(len(base))]; gross=sum(abs(v) for v in x) or 1; x=[v/gross for v in x]
            kv=[_rbf(x,xi) for xi in xs]; mu=_dot(kv,alpha); v=_solve(k,kv); sd=math.sqrt(max(1e-12,1-_dot(kv,v))); z=(mu-best)/sd
            ei=(mu-best)*_cdf(z)+sd*_pdf(z)
            if ei>best_ei: best_ei=ei; chosen=x
        xs.append(chosen); ys.append(score(chosen))
    i=max(range(len(ys)),key=ys.__getitem__)
    return tuple(xs[i]),ys[i]

def _bucket_returns(ticks,minutes):
    ns=minutes*60*1_000_000_000; closes={}; spreads={}
    for t in ticks:
        ts=int(t.ts_event); bucket=(ts//ns)*ns; bid=_f(t.bid_price); ask=_f(t.ask_price)
        closes[bucket]=(bid+ask)/2; spreads[bucket]=ask-bid
    keys=sorted(closes); out={}; prev=None
    for k in keys:
        px=closes[k]
        if prev is not None and prev>0: out[k]=px/prev-1
        prev=px
    return out,(mean(spreads.values()) if spreads else None),len(closes)

def _aligned(series_by_symbol):
    keys=set.intersection(*(set(v) for v in series_by_symbol.values()))
    ks=sorted(keys); syms=list(series_by_symbol)
    return syms,ks,[[series_by_symbol[s][k] for k in ks] for s in syms]

def _wfo(rows,*,train=240,test=80,step=80,seed=20260921,max_folds=12):
    n=len(rows[0]); folds=[]; start=max(0,n-(train+test+(max_folds-1)*step)); fold=0
    while start+train+test<=n and fold<max_folds:
        tr=[r[start:start+train] for r in rows]; te=[r[start+train:start+train+test] for r in rows]
        weights,obj=_optimize(tr,[1/len(rows)]*len(rows),seed=seed+fold)
        rs=_basket(te,weights); p=_risk_packet(rs,paths=160,horizon=min(40,test),seed=seed+1000+fold)
        folds.append({"fold":fold,"train_start":start,"train_end":start+train,"test_start":start+train,"test_end":start+train+test,
                      "weights":weights,"optimizer_objective":obj,"oos_mean":mean(rs),"oos_vol":pstdev(rs),
                      "oos_sharpe_like":mean(rs)/max(pstdev(rs),1e-12),"oos_maxdd":_maxdd(rs),**p})
        start+=step; fold+=1
    if not folds: raise ValueError("insufficient aligned history for WFO")
    return folds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True); ap.add_argument("--symbols",nargs="+",required=True); ap.add_argument("--timeframes",nargs="+",required=True)
    ap.add_argument("--experiment-id",required=True); ap.add_argument("--raw-bidask-only",action="store_true")
    ap.add_argument("--train-size",type=int,default=240); ap.add_argument("--test-size",type=int,default=80); ap.add_argument("--max-folds",type=int,default=12)
    args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit("RAW_BIDASK_ONLY_REQUIRED")
    cp=Path(args.catalog); manifest=json.loads((cp/"catalog_manifest.json").read_text(encoding="utf-8")); cat=ParquetDataCatalog(str(cp))
    inst={x.id.symbol.value.replace("/",""):x for x in cat.instruments()}; raw_counts={}; tick_cache={}
    for s in args.symbols:
        if s not in inst: raise SystemExit(f"CATALOG_INSTRUMENT_MISSING:{s}")
        ticks=cat.query_quote_ticks(identifiers=[inst[s].id.value])
        if not ticks: raise SystemExit(f"RAW_QUOTETICK_MISSING:{s}")
        tick_cache[s]=ticks; raw_counts[s]=len(ticks)
    tf_reports={}; fold_rows=[]; spread_stats={}
    for tf in args.timeframes:
        if tf not in TF_MIN: raise SystemExit(f"UNSUPPORTED_TF:{tf}")
        by={}; spread_stats[tf]={}
        for s in args.symbols:
            by[s],avg_spread,buckets=_bucket_returns(tick_cache[s],TF_MIN[tf])
            spread_stats[tf][s]={"avg_observed_spread":avg_spread,"raw_derived_buckets":buckets}
        syms,keys,rows=_aligned(by)
        folds=_wfo(rows,train=args.train_size,test=args.test_size,step=args.test_size,seed=20260921+TF_MIN[tf],max_folds=args.max_folds)
        for f in folds: fold_rows.append({"timeframe":tf,**f})
        tf_reports[tf]={"symbols":syms,"aligned_return_points":len(keys),"fold_count":len(folds),
                        "mean_oos_sharpe_like":mean(f["oos_sharpe_like"] for f in folds),
                        "worst_oos_maxdd_pct":100*max(f["oos_maxdd"] for f in folds),
                        "mean_oos_cvar_pct":100*mean(f["cvar_loss"] for f in folds),
                        "mean_tail_probability":mean(f["first_passage_tail"] for f in folds),
                        "mean_recovery_probability":mean(f["first_passage_recovery"] for f in folds),
                        "mean_ae_tightening_cap":mean(f["ae_tightening_cap"] for f in folds)}
    out=Path("results/ae-bt")/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    summary={"schema":SCHEMA,"verification_level":"NAUTILUS_RAW_RESEARCH_WFO",
             "engine":"NautilusTrader ParquetDataCatalog/QuoteTick + chronological AMOS WFO research evaluator",
             "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"data_kind":"RAW_BIDASK QuoteTick","ohlc_resample_used":False,
             "execution_model":"NO_ORDER_EXECUTION_IN_THIS_RESEARCH_WFO","symbols":args.symbols,"timeframes":args.timeframes,
             "period":{"start":manifest.get("start"),"days":manifest.get("days"),"end_exclusive":manifest.get("end_exclusive")},
             "raw_tick_counts":raw_counts,"spread_stats":spread_stats,"timeframe_reports":tf_reports,
             "claim_boundary":{"gaussian_baseline":"SOURCE_DEFINED_ARCHITECTURE",
               "regime_fat_tail_first_passage_cvar_wasserstein":"AMOS_HYPOTHESIS_UNTIL_VALIDATED",
               "crystal_ball":"RESEARCH_CONTEXT_ONLY","ae_dd_supervisor":"TIGHTENING_CAP_ONLY"},
             "production_weighting_allowed":False,"execution_allowed":False,
             "limitations":["This validates research transforms on raw QuoteTick-derived returns; it is not a broker-order strategy BT and does not report WR/PF/Net as trading KPIs.",
               "Scenario shares are not calibrated future probabilities.",
               "Wasserstein stressed CVaR is the documented AMOS diagnostic proxy CVaR+radius, not a complete DRO solver."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"catalog_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    fields=["timeframe","fold","train_start","train_end","test_start","test_end","weights","optimizer_objective","oos_mean","oos_vol","oos_sharpe_like","oos_maxdd","regime","volatility_ratio","first_passage_recovery","first_passage_tail","first_passage_unresolved","cvar_loss","wasserstein_stressed_cvar","ae_tightening_cap","crystal_ball_context"]
    with (out/"walk_forward_folds.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for row in fold_rows:
            row=dict(row); row["weights"]=json.dumps(row["weights"]); w.writerow({k:row.get(k) for k in fields})
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
