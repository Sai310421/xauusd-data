from __future__ import annotations
import argparse, json, sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import g75_negative_memory_v5_2_hydra66 as base

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


class OnlineLossGate:
    def __init__(self, min_labels=240, retrain_every=80):
        self.min_labels=min_labels; self.retrain_every=retrain_every
        self.resolved=[]; self.model=None; self.threshold=None; self.last_fit_n=0; self.model_version=0
        self.live=None; self.live_records=[]; self.decisions=[]; self.fit_history=[]

    def _fit_if_needed(self):
        n=len(self.resolved)
        if n < self.min_labels: return
        if self.model is not None and n-self.last_fit_n < self.retrain_every: return
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X=np.vstack([r['features'] for r in self.resolved]).astype(np.float32)
        y=np.asarray([1 if r['pnl'] < 0 else 0 for r in self.resolved], int)
        pnl=np.asarray([r['pnl'] for r in self.resolved], float)
        if len(np.unique(y)) < 2: return
        m=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200,class_weight='balanced',C=.5))
        m.fit(X,y); p=m.predict_proba(X)[:,1]
        th,wr,la,cov=base.choose_threshold(y,p,pnl,min_win_ret=.55,min_cov=.20)
        self.model=m; self.threshold=float(th); self.last_fit_n=n; self.model_version += 1
        self.fit_history.append({'model_version':self.model_version,'resolved_n':n,'threshold':float(th),'train_win_retention':float(wr),'train_loss_avoidance':float(la),'train_coverage':float(cov)})

    def on_shadow_resolved(self, rec):
        self.resolved.append(rec); self._fit_if_needed()

    def decide(self, candidate):
        self._fit_if_needed()
        if self.model is None:
            d={'entry_ts':candidate['entry_ts'],'decision':'WARMUP','p_loss':None,'threshold':None,'model_version':self.model_version}
        else:
            p=float(self.model.predict_proba(candidate['features'].reshape(1,-1).astype(np.float32))[0,1])
            d={'entry_ts':candidate['entry_ts'],'decision':'PASS' if p < self.threshold else 'REJECT','p_loss':p,'threshold':self.threshold,'model_version':self.model_version}
        self.decisions.append(d); return d

    def open_live(self, c):
        if self.live is not None: return False
        s=int(c['side']); entry=float(c['ask'] if s>0 else c['bid'])
        self.live={'entry_ts':c['entry_ts'],'side':s,'entries':[entry],'last_add':entry,'extreme':float(c['bid'] if s>0 else c['ask']),'regime':c['regime'],'p_loss':self.decisions[-1]['p_loss'],'threshold':self.decisions[-1]['threshold'],'model_version':self.decisions[-1]['model_version']}
        return True

    def update_live(self, bid, ask):
        if self.live is None: return
        s=self.live['side']; px=bid if s>0 else ask
        self.live['extreme']=max(self.live['extreme'],px) if s>0 else min(self.live['extreme'],px)
        while len(self.live['entries']) < 10:
            tgt=self.live['last_add'] + s*.025
            if not (px>=tgt if s>0 else px<=tgt): break
            self.live['entries'].append(ask if s>0 else bid); self.live['last_add']=tgt

    def close_live_if_open(self, ts,bid,ask,reason):
        if self.live is None: return 0
        s=self.live['side']; px=bid if s>0 else ask; pnl=sum((px-e)*s for e in self.live['entries'])
        r=dict(self.live); r.update(exit_ts=int(ts),pnl=float(pnl),win=int(pnl>0),reason=reason,layers=len(self.live['entries']))
        self.live_records.append(r); self.live=None; return 1


class ShadowCandidateEngine:
    def __init__(self, gate, live_gate):
        self.gate=gate; self.live_gate=live_gate; self.trigger=.12; self.add=.025; self.rev=.20; self.need=.025 if gate=='C1' else .05; self.pullback=.04
        self.bucket=None; self.anchor=None; self.started=False; self.pending=0; self.tpx=None; self.pext=None; self.mids=deque(maxlen=16)
        self.active=False; self.side=0; self.entries=[]; self.last_add=None; self.extreme=None; self.trade=None
        self.audit={k:0 for k in ['trigger_count','continuation_count','feature_ready_count','feature_not_ready_count','hydra_pass_count','hydra_reject_count','warmup_reject_count','live_entry_count','live_exit_count','shadow_entry_count','shadow_exit_count','entry_block_other_count']}

    def _close_shadow(self,ts,bid,ask,reason):
        if not self.active:return
        px=bid if self.side>0 else ask; pnl=sum((px-e)*self.side for e in self.entries)
        r=dict(self.trade); r.update(exit_ts=int(ts),pnl=float(pnl),win=int(pnl>0),reason=reason,layers=len(self.entries))
        self.audit['shadow_exit_count']+=1; self.live_gate.on_shadow_resolved(r)
        self.active=False; self.side=0; self.entries=[]; self.trade=None

    def _candidate(self,ts,bid,ask,feat,reg,s):
        self.audit['continuation_count']+=1
        if feat is None:
            self.audit['feature_not_ready_count']+=1; return
        self.audit['feature_ready_count']+=1
        c={'entry_ts':int(ts),'side':int(s),'features':np.asarray(feat,np.float32).copy(),'regime':str(reg),'bid':float(bid),'ask':float(ask)}
        d=self.live_gate.decide(c)
        if d['decision']=='PASS':
            self.audit['hydra_pass_count']+=1
            if self.live_gate.open_live(c): self.audit['live_entry_count']+=1
            else: self.audit['entry_block_other_count']+=1
        elif d['decision']=='WARMUP': self.audit['warmup_reject_count']+=1
        else: self.audit['hydra_reject_count']+=1
        entry=ask if s>0 else bid; self.side=s; self.active=True; self.entries=[entry]; self.last_add=entry; self.extreme=bid if s>0 else ask
        self.trade={'entry_ts':int(ts),'side':int(s),'features':c['features'],'regime':str(reg),'model_decision':d['decision'],'p_loss':d['p_loss'],'threshold':d['threshold'],'model_version':d['model_version']}
        self.audit['shadow_entry_count']+=1

    def update(self,ts,bid,ask,feat,reg):
        mid=(bid+ask)/2; self.mids.append(mid); b=ts//60
        if self.bucket is None: self.bucket=b; self.anchor=mid
        elif b!=self.bucket:
            if self.active:
                px=bid if self.side>0 else ask; rev=px<=self.extreme-self.rev if self.side>0 else px>=self.extreme+self.rev
                if rev:
                    self._close_shadow(ts,bid,ask,'REV'); self.audit['live_exit_count']+=self.live_gate.close_live_if_open(ts,bid,ask,'REV')
            self.bucket=b; self.anchor=self.mids[-2] if len(self.mids)>1 else mid; self.started=False; self.pending=0
        if not self.active and not self.started:
            if self.pending==0:
                up=mid>=self.anchor+self.trigger; dn=mid<=self.anchor-self.trigger
                if up or dn:
                    self.pending=1 if up else -1; self.tpx=mid; self.pext=mid; self.audit['trigger_count']+=1
            else:
                s=self.pending; self.pext=max(self.pext,mid) if s>0 else min(self.pext,mid); ext=(self.pext-self.tpx)*s; pb=(self.pext-mid)*s
                if pb>self.pullback: self.started=True; self.pending=0
                elif ext>=self.need and (self.gate!='C3' or (len(self.mids)>=6 and (self.mids[-1]-self.mids[-6])*s>0)):
                    self._candidate(ts,bid,ask,feat,reg,s); self.started=True; self.pending=0
        if self.active:
            px=bid if self.side>0 else ask; self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
            while len(self.entries)<10:
                tgt=self.last_add+self.side*self.add
                if not (px>=tgt if self.side>0 else px<=tgt): break
                self.entries.append(ask if self.side>0 else bid); self.last_add=tgt
        self.live_gate.update_live(bid,ask)

    def finish(self,ts,bid,ask):
        if self.active:self._close_shadow(ts,bid,ask,'EOD')
        self.audit['live_exit_count']+=self.live_gate.close_live_if_open(ts,bid,ask,'EOD')


def summary(records,name):
    if not records:return {'name':name,'N':0,'WR_pct':0.0,'PF':0.0,'return_pct_on_1000':0.0,'max_DD_pct':0.0,'gross_usd':0.0}
    p=np.asarray([r['pnl'] for r in records],float)
    return {'name':name,'N':len(p),'WR_pct':100*float(np.mean(p>0)),'PF':float(base.pf(p)),'return_pct_on_1000':float(p.sum()/10),'max_DD_pct':float(base.maxdd(p)),'gross_usd':float(p.sum())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--gate',choices=['C1','C2','C3'],required=True); ap.add_argument('--raw-bidask-only',action='store_true')
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    cat=ParquetDataCatalog(str(Path(a.catalog))); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'); ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    fe=base.FeatureEngine(); lg=OnlineLossGate(); eng=ShadowCandidateEngine(a.gate,lg); last=None
    for t in ticks:
        bid=float(t.bid_price.as_double()); ask=float(t.ask_price.as_double()); ts=int(t.ts_event)//1_000_000_000
        fe.update(ts,(bid+ask)/2,ask-bid); eng.update(ts,bid,ask,fe.latest,fe.regime()); last=(ts,bid,ask)
    if last: eng.finish(*last)
    audit=dict(eng.audit); audit['resolved_shadow_labels']=len(lg.resolved); audit['model_versions']=lg.model_version; audit['hydra_pass_minus_entry']=audit['hydra_pass_count']-audit['live_entry_count']
    audit['entry_rate_from_trigger']=audit['live_entry_count']/audit['trigger_count'] if audit['trigger_count'] else 0.0
    audit['entry_rate_from_hydra_pass']=audit['live_entry_count']/audit['hydra_pass_count'] if audit['hydra_pass_count'] else 0.0
    audit['invariant_chain_ok']=bool(audit['live_entry_count']<=audit['hydra_pass_count']<=audit['feature_ready_count']<=audit['continuation_count']<=audit['trigger_count'])
    audit['pass_entry_parity_ok']=bool(audit['hydra_pass_minus_entry']==0)
    audit['trigger_nonzero_entry_zero_fail']=bool(audit['trigger_count']>0 and audit['live_entry_count']==0)
    resolved=[r for r in lg.resolved if r.get('model_decision') in ('PASS','REJECT')]; losses=[r for r in resolved if r['pnl']<0]; wins=[r for r in resolved if r['pnl']>0]
    audit['loss_avoidance_resolved_pct']=100*sum(r['model_decision']=='REJECT' for r in losses)/len(losses) if losses else 0.0
    audit['win_retention_resolved_pct']=100*sum(r['model_decision']=='PASS' for r in wins)/len(wins) if wins else 0.0
    failures=[]
    if audit['trigger_nonzero_entry_zero_fail']: failures.append('TRIGGER_NONZERO_ENTRY_ZERO')
    if not audit['invariant_chain_ok']: failures.append('COUNT_CHAIN_INVARIANT_BROKEN')
    if not audit['pass_entry_parity_ok']: failures.append('HYDRA_PASS_WITHOUT_ENTRY')
    out=Path('results/ae-bt')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    payload={'version':'v5.3-online-audit','gate':a.gate,'execution_contract':'Shadow candidates are labeled causally; Hydra decision occurs before LIVE entry','audit':audit,'shadow_summary':summary(lg.resolved,'SHADOW_ALL_CANDIDATES'),'live_summary':summary(lg.live_records,'LIVE_HYDRA66_GATED'),'fit_history':lg.fit_history,'audit_failures':failures}
    (out/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); pd.DataFrame(lg.decisions).to_csv(out/'decisions.csv',index=False)
    if lg.live_records: pd.DataFrame(lg.live_records).to_csv(out/'live_trades.csv',index=False)
    print(json.dumps(payload,indent=2))
    if failures: raise SystemExit('AUDIT_FAIL: '+','.join(failures))

if __name__=='__main__': main()
