# USDJPY Institutional Risk AI

USDJPY、Gold、米国株、Russell 2000 の市場リスクを H1 単位で監視し、個人向けマクロヘッジ判断を支援するための研究・運用基盤です。

このコンポーネントは**発注しません**。校正済み確率、レジーム、説明可能な要因、ドリフト状態だけを出力し、執行側とはアダプター契約で分離します。

## 現在の到達点

- OANDA USDJPY H1 Mid と Databento Continuous Front H1 を正本として固定
- 限月ロールを自前計算・バックアジャストせず、ロールイベントを特徴量とラベルから隔離
- USDJPY急落、Gold上昇、米株急落の LightGBM 学習・時系列校正・Walk Forward
- PSI、Feature Integrity、Manifest Integrity、Probability Ordering による fail-closed 判定
- Discordレポート、SHAP Top Drivers、予測履歴
- OCI Object Storage の不変スナップショットで、GitHub Actions の実行機終了後も状態を復元
- Portfolio Hedge Engine は助言専用。注文・ロット計算・ブローカー接続は含まない

実データを使う本番稼働には、別途 Databento と OANDA の利用契約・Secrets設定が必要です。

## 構成

```text
usdjpy-risk-ai/
├── src/
│   ├── collect.py / collect_multiasset.py
│   ├── features.py / asset_features.py
│   ├── train.py / train_assets.py / walkforward.py
│   ├── predict.py / portfolio_predict.py / portfolio_engine.py
│   ├── feature_store.py / storage.py / production_gate.py
│   └── report.py
├── tests/
├── config.yaml
├── requirements.txt
└── docs/
    └── PRODUCTION_PRECHECK_JA.md
```

## 無料の Fixture 検証

```bash
cd usdjpy-risk-ai
python tests/fixtures/make_fixture.py --output /tmp/raw_fixture.parquet
python -m src.collect --input /tmp/raw_fixture.parquet --output data/raw
python -m src.features --input data/raw --output data/features
pytest -q
ruff check .
```

この経路は外部データAPIや OCI を呼びません。

## 本番前の設定

GitHub Secrets に次を設定します。

```text
DATABENTO_API_KEY
OANDA_API_TOKEN
OCI_OBJECT_STORAGE_BUCKET
OCI_OBJECT_STORAGE_NAMESPACE
OCI_USER_OCID
OCI_TENANCY_OCID
OCI_FINGERPRINT
OCI_PRIVATE_KEY
```

設定後の順序は以下です。

1. `USDJPY Risk AI Production Collect` で履歴を収集し、OCIへ不変保存
2. `USDJPY Risk AI Production Train` で復元・特徴量生成・Walk Forward・学習
3. Quality Gateを通過した候補だけを production manifest に昇格
4. `USDJPY Risk AI Production Predict` で復元・検証・推論・通知
5. 30日Shadow運用の全条件を満たしてから本格監視へ進む

詳細は [本番前チェックリスト](docs/PRODUCTION_PRECHECK_JA.md) を参照してください。

## 安全原則

- 価格系列・特徴量・モデルはハッシュ検証される。
- データ鮮度、必須特徴量、Manifest、Lead-Lag、確率順序の異常時は推論停止。
- `portfolio_risk` は危機度であり、ヘッジ取引の市場コスト `risk_cost_t` と混在させない。
- 投資助言・売買執行ではなく、リスク判断支援として運用する。
