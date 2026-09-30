"""Dispatch only reviewed, source-bound Nautilus ports. Never execute uploaded source."""
import hashlib, json, os, re, shutil, subprocess, sys
from datetime import date
from pathlib import Path
ROOT=Path.cwd()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(req,registry):
    if req.get('schema')!='amos.nautilus.request.v1': raise ValueError('INVALID_REQUEST_SCHEMA')
    rid=req.get('request_id','')
    if not re.fullmatch(r'[-a-zA-Z0-9]{8,80}',rid): raise ValueError('INVALID_REQUEST_ID')
    source=req.get('source_sha256','')
    if not re.fullmatch(r'[a-f0-9]{64}',source): raise ValueError('INVALID_SOURCE_HASH')
    cfgtext=req.get('config_json','')
    if hashlib.sha256(cfgtext.encode()).hexdigest()!=req.get('config_sha256'): raise ValueError('CONFIG_HASH_MISMATCH')
    cfg=json.loads(cfgtext)
    if cfg.get('source_sha256')!=source or cfg.get('data_kind')!='RAW_BIDASK' or cfg.get('synthetic') is not False: raise ValueError('RAW_SOURCE_CONTRACT_MISMATCH')
    start,end=date.fromisoformat(cfg['start']),date.fromisoformat(cfg['end']); days=(end-start).days
    if not 1<=days<=90 or end>date.today(): raise ValueError('INVALID_DATA_WINDOW_MAX_90_DAYS')
    if cfg.get('symbol') not in ['XAUUSD','EURUSD','GBPUSD','USDJPY','AUDUSD','USDCHF']: raise ValueError('UNSUPPORTED_SYMBOL')
    if not 0<float(cfg.get('initial',0))<=10000000 or not 0<float(cfg.get('leverage',0))<=2000: raise ValueError('INVALID_ACCOUNT_CONFIG')
    entry=registry.get('strategies',{}).get(source)
    if not entry or entry.get('port_review')!='APPROVED': raise ValueError('SOURCE_NOT_REGISTERED')
    for name in ['strategy_path','adapter_path']:
        path=ROOT/entry[name]
        if not path.resolve().is_relative_to((ROOT/'research').resolve()) or path.suffix!='.py': raise ValueError('UNTRUSTED_PORT_PATH')
        if sha(path)!=entry[name.replace('_path','_sha256')]: raise ValueError('REGISTERED_PORT_HASH_MISMATCH')
    if cfg['symbol'] not in entry.get('symbols',[]): raise ValueError('PORT_SYMBOL_NOT_APPROVED')
    return cfg,entry,days

def main():
    req=json.loads(os.environ['BT_REQUEST_JSON']);out=Path('site-bt-output');out.mkdir(exist_ok=True)
    result={'status':'BLOCKED','request_id':req.get('request_id'),'source_sha256':req.get('source_sha256')}
    try:
        reg=json.loads(Path('research/amos_site_bt/registry.json').read_text());cfg,entry,days=validate(req,reg)
        out.joinpath('request.json').write_text(json.dumps(req))
        env=os.environ.copy();env.update({'RAW_START':cfg['start'],'RAW_DAYS':str(days),'RAW_CATALOG':'catalog/raw_bidask','RAW_WORKERS':'8'})
        catalog=Path('catalog/raw_bidask');manifest=catalog/'catalog_manifest.json'
        if catalog.exists():
            try:
                old=json.loads(manifest.read_text())
                matches=old.get('status')=='COMPLETE' and old.get('start')==cfg['start'] and old.get('end_exclusive')==cfg['end']
            except Exception: matches=False
            if not matches: shutil.rmtree(catalog)
        # Existing repository preflight restores/bootstrap/retries without OHLC fallback.
        subprocess.run(['bash','research/ensure_raw_bidask_catalog.sh'],env=env,check=True,timeout=5400)
        subprocess.run([sys.executable,'research/raw_bidask_catalog_guard.py','--catalog','catalog/raw_bidask'],env=env,check=True,timeout=300)
        data=json.loads(manifest.read_text())
        if data.get('start')!=cfg['start'] or data.get('end_exclusive')!=cfg['end'] or data.get('status')!='COMPLETE': raise ValueError('RAW_DATA_WINDOW_MISMATCH')
        subprocess.run([sys.executable,entry['adapter_path'],'--request',str(out/'request.json'),'--catalog','catalog/raw_bidask','--output',str(out/'result.json')],env=env,check=True,timeout=5400)
        result=json.loads((out/'result.json').read_text());m=result.get('manifest',{})
        for key,value in [('source_sha256',req['source_sha256']),('strategy_sha256',entry['strategy_sha256']),('config_sha256',req['config_sha256']),('status','COMPLETED'),('port_review','APPROVED'),('data_kind','RAW_BIDASK'),('synthetic',False)]:
            if m.get(key)!=value: raise ValueError('ADAPTER_EVIDENCE_MISMATCH_'+key)
        if result.get('schema')!='amos.nautilus.kpi.v1' or not result.get('trades') or not result.get('equity'): raise ValueError('ADAPTER_RESULT_INCOMPLETE')
        import nautilus_trader
        if m.get('nautilus_version')!=nautilus_trader.__version__ or m.get('dataset_sha256')!=data.get('catalog_sha256') or m.get('ended_flat') is not True: raise ValueError('NATIVE_DATASET_OR_LIQUIDATION_EVIDENCE_MISMATCH')
        # Pin transport evidence to the actual Actions run. UI independently recomputes all KPI.
        m.update({'run_id':os.environ['GITHUB_RUN_ID'],'git_sha':os.environ['GITHUB_SHA'],'workflow':'amos-site-nautilus.yml','artifact_ref':'amos-bt-'+req['request_id']})
        result['request_id']=req['request_id'];result['status']='COMPLETED'
    except Exception as exc:
        result={'status':'BLOCKED','request_id':req.get('request_id'),'source_sha256':req.get('source_sha256'),'reason':str(exc)[:400]}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False))
    print(json.dumps({'status':result['status'],'request_id':req.get('request_id'),'reason':result.get('reason')}))
    return 0 if result['status']=='COMPLETED' else 1
if __name__=='__main__':sys.exit(main())
