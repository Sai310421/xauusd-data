from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'config' / 'arem_multi_myfxbook_foundation.json'
OUT = ROOT / 'artifacts' / 'arem_multi_edge_foundation'
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.loads(CFG.read_text(encoding='utf-8'))

required_top = ['version', 'architecture', 'edge_families', 'validation']
missing = [k for k in required_top if k not in cfg]
if missing:
    raise SystemExit(f'Missing foundation keys: {missing}')

# Every source must remain identifiable; never average away source provenance.
sources = cfg.get('sources', [])
ids = [s.get('strategy_id') for s in sources]
if len(ids) != len(set(ids)):
    raise SystemExit('Duplicate strategy_id detected')

manifest = {
    'status': 'FOUNDATION_SCHEMA_VALID',
    'version': cfg['version'],
    'registered_sources': len(sources),
    'strategy_ids': ids,
    'edge_families': cfg['edge_families'],
    'validation': cfg['validation'],
    'principles': [
        'preserve_strategy_provenance',
        'route_by_regime_not_naive_average',
        'separate_empirical_edge_from_math_edge',
        'require_time_strategy_regime_oos',
        'risk_supervisor_remains_deterministic'
    ]
}
(OUT / 'foundation_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
