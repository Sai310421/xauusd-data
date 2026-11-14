# USDJPY Institutional Risk AI — 本番前チェックリスト

このファイルは、本番データを投入する直前に一度だけ確認するための運用チェックリストです。未達項目がある状態では production 実行を開始しません。

## 1. データ契約

- [ ] Databento API Key が有効で、固定 Dataset / Schema にアクセスできる
- [ ] OANDA Practice または本番 Token が有効で、`USD_JPY` の H1 Mid (`price=M`) を取得できる
- [ ] 先物は provider continuous-front のみを利用し、自前ロール・バックアジャストをしていない
- [ ] Databento と OANDA のタイムゾーンが UTC / completed H1 bar で整合している

## 2. GitHub Secrets と OCI

- [ ] `DATABENTO_API_KEY`
- [ ] `OANDA_API_TOKEN`
- [ ] `OCI_OBJECT_STORAGE_BUCKET`
- [ ] `OCI_OBJECT_STORAGE_NAMESPACE`
- [ ] `OCI_USER_OCID`
- [ ] `OCI_TENANCY_OCID`
- [ ] `OCI_FINGERPRINT`
- [ ] `OCI_PRIVATE_KEY`
- [ ] OCI バケットに prefix `usdjpy-risk-ai/` への Put / Get / List 権限がある
- [ ] Secretsはログ・レポート・Git履歴に一切出力されない

## 3. 最初の実データ実行

1. Collect workflow を小さい期間で手動起動する。
2. OCI に `state/raw/` と `state/multiasset/` の manifest が作成されることを確認する。
3. 同じ workflow を再実行し、既存スナップショットがハッシュ検証付きで復元されることを確認する。
4. Train workflow を実行する。
5. 以下を満たす候補だけを production manifest に昇格する。

```text
AUC_ROC >= 0.60
PR_AUC > Baseline
Brier Score <= 0.25
ECE <= 0.05
対象ラベルの学習陽性数 >= 50
全Walk Forward Fold合計の陽性数 >= 100
各Foldの陽性数 >= 20
Feature / Manifest / Future Leakage / Probability Ordering / OCI Upload が全Pass
```

満たさない場合は、新規 production を作らず既存 production を維持する。

## 4. Predict の停止条件

以下のどれかで推論を停止し Discord に理由を通知する。

- 最新バーが90分超過
- 必須特徴量の欠損率が5%超過
- Feature Schema・順序・型が不一致
- Manifest / モデル / Calibrator / ハッシュが無効
- Lead-Lag の観測不足、ゼロ分散、NaN、例外
- `prob_20 <= prob_10 <= prob_05` を満たさない
- OCI取得失敗
- PSIが Critical

## 5. Shadow 運用からの移行

30日間、少なくとも以下を満たしてから本格監視へ移行する。

- [ ] 重大エラー 0件
- [ ] 停止条件の誤発火 0件
- [ ] Prediction Log 欠損 0件
- [ ] 通知重複 0件
- [ ] Manifest破損 0件
- [ ] Probability Ordering違反 0件

## 完了の定義

このチェックリストの完了は「本番稼働の許可」ではありません。実データの収集、学習、品質ゲート、30日Shadow運用がすべて成功した時点で初めて監視運用へ進めます。
