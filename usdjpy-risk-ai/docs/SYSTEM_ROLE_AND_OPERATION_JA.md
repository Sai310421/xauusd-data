# USDJPY Institutional Risk AI — 役割と動作

## 1. このシステムの役割

これは自動売買システムではありません。

USDJPY、Gold、米国株、小型株の市場ストレスを毎時観測し、保有ポートフォリオに対する**リスク判断材料**を出すための監視・意思決定支援基盤です。

出力は確率、レジーム、主要因、ポートフォリオ危険度、推奨ヘッジ行動であり、ブローカーへの発注・ロット計算・注文管理は行いません。

## 2. システム構成

| 層 | 対象 | 主な出力 |
|---|---|---|
| USDJPY Risk AI | 円先物、VIX、株価指数、債券、USDJPY | 24時間以内のUSDJPY急落確率 |
| Gold AI | 金先物、ドル指数、金利、VIX、株式、RTY | Gold上昇確率 |
| US Stock AI | VIX、ES、NQ、RTY、債券、ドル | 米株急落確率 |
| Russell 2000 Risk Engine | RTY、ES、NQ | SmallCap Stress、Breadth Divergence |
| Portfolio Hedge Engine | 上記の推論、資産配分、PSI | Portfolio Risk、Regime、Suggested Hedge |

## 3. 入力データの固定契約

| データ | 正本 | ルール |
|---|---|---|
| USDJPY | OANDA H1 Mid Price | Bid・Ask・他ブローカー価格をラベルや学習価格へ混在させない |
| 6J / ES / NQ / ZN / RTY / GC | Databento Continuous Front H1 | Providerの連続限月を使用。自前ロール・Back Adjustは禁止 |
| VX | Databento 固定Dataset / H1 Schema | 設定外Schemaへの自動切替は禁止 |

すべてUTCの**確定済みH1バー**だけを使います。途中のバーや将来バーは特徴量へ入れません。

## 4. 毎時の動作

1. 最新の確定H1データを取得する。
2. OCI Object Storageから前回までのrawデータ、特徴量、モデルをハッシュ検証付きで復元する。
3. 新しいデータを月単位Parquetへ統合する。
4. 円フロー、VIXフロー、株式ストレス、JPY Divergence、Lead-Lag、小型株ストレスなどを計算する。
5. 学習済みLightGBMと校正器で、各対象の確率を計算する。
6. SHAP絶対値の上位3件をTop Driversとして出す。
7. Portfolio Hedge Engineが確率・資産配分・PSIを統合し、危険度と助言を出す。
8. JSON、HTML、Prediction Log、Discord通知を出力する。
9. 新たな状態をOCIへ不変スナップショットとして保存する。

## 5. 出力例

```text
USDJPY Crash Model

-0.5% within 24h: 72%
-1.0% within 24h: 45%
-2.0% within 24h: 18%

Regime: RISK_OFF
Top Drivers: Flow_6J / JPY Divergence / Flow_VX
```

```text
Portfolio Hedge Engine

Gold Probability: 74%
Stock Crash Probability: 68%
SmallCap Stress: 0.82
Portfolio Risk: 77 / 100
Portfolio Regime: HIGH
Suggested Hedge: Reduce Equity / Increase Gold
```

## 6. 安全停止の仕組み

次のいずれかが起きた場合、推論を止め、結果を出さずにDiscordへ停止理由を通知します。

- 最新バーが90分を超えて古い
- 必須特徴量の欠損率が5%を超える
- 学習時と推論時のFeature Schema、順序、型が不一致
- Manifest、モデル、Calibrator、ハッシュが欠損または不正
- Lead-Lagの観測数不足、ゼロ分散、NaN、例外
- `prob_20 <= prob_10 <= prob_05` の確率順序違反
- OCIからの取得失敗
- PSIがCritical水準

この仕組みは「壊れたデータやモデルで、それらしく見える予測を出さない」ためのものです。

## 7. AMOS RAGEとの境界

本システムはセンサーです。市場の危機度、確率、説明可能な要因だけを渡します。

AMOS RAGE DCHはアクチュエータです。Raw Tick、証拠金、実行コスト、保護資本、執行ロットを扱います。

両者を接続する際は、以下を厳守します。

- `portfolio_risk` は「守るべきポートフォリオの危機度」
- `risk_cost_t` は「ヘッジ取引自体の市場コスト」
- 二つを同じ変数や抑制分母に混在させない
- 危機度の上昇は、ヘッジ強度を上げる方向にだけ作用させる

この分離により、危機度が高い局面でヘッジ比率が逆に小さくなる論理逆転を防ぎます。

## 8. 完了と未証明の範囲

実装基盤としては、データ収集、特徴量、学習、推論、安全停止、OCI永続化、GitHub Actionsの経路を備えています。

ただし、次は実データでしか証明できません。

- Databento / OANDA / OCIの本番権限とデータ形式の整合
- 学習モデルが品質ゲートを通過すること
- 実市場でのPSI、データ鮮度、停止条件の挙動
- 30日Shadow運用での無重大エラー・無誤発火
- ヘッジ助言が実際の損失低減につながること

したがって、この文書は機能説明であり、収益性またはヘッジ効果の保証ではありません。
