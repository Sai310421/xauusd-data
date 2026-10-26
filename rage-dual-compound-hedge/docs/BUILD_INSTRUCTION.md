# AMOS RAGE Dual Compound Hedge Research Laboratory
## 完全検証指示書・必要事項一式（Build実行用）

**作成日**: 2026-10-26  
**バージョン**: 1.1  
**対象**: Build（OpenCode / 開発エージェント / AMOS Public CI）  
**Live Trading**: OFF（研究・Shadow・MT5 DEMO限定）  
**原則**: 証拠のない成功報告禁止。仮説を達成したように見せない。

---

## 0. 本プロジェクトの最重要趣旨

保有資産の収益性と、その保有資産を守るヘッジ側の収益性を、両方とも高める研究システムである。

通常のヘッジは保険料・機会損失が発生する。  
RAGEではヘッジ側にも独立した収益エンジンを持たせ、保護能力を維持しながらその資本自体を複利成長させる。

### 目標
- **MAIN ENGINE**: 保有資産を積極運用。月利10%以上の複利成長を目標。初期検証対象はXAUUSD。既存AMOS収益戦略を再利用。
- **HEDGE ENGINE**: MAINの損失を補填しつつ、自身も月利10%の複利成長を目標。保護資本を無条件に収益運用へ投入しない。
- **統合**: 上昇・下落・レンジ・急変動のいずれでもMAINとHEDGEが互いの損失を補完し、総資産のプラス成長を目指す。両エンジンの同時損失を防ぐ数学的仕組みを重視。

これは**未証明の研究仮説**である。達成を前提に結果を作ってはならない。成立条件・不成立条件・必要資本・取引費用・市場制約を数理的に明らかにし、実データで再現可能性を検証する。

---

## 1. 研究仮説（RAGE-DCH-H1）

| ID | 内容 |
|----|------|
| H1-A | MAINの月間純収益率10%以上 |
| H1-B | HEDGEの月間純収益率10%以上 |
| H1-C | MAIN損失時にHEDGEが補填する |
| H1-D | HEDGE損失時にMAINが補填する |
| H1-E | 両エンジンの同時損失を抑制する |
| H1-F | 統合資産の複利成長を実現する |
| H1-G | 上記の性質がOOS/WFO/Monte Carloでも維持される |

月利10%は期待値だけでなく、実現月次リターンの分布・未達月数・連続未達期間・最悪月・中央値・幾何平均成長率で評価する。  
「月利10%の達成」と「すべての月で10%以上を保証」は区別する。

### 数学的基本構造
- MAIN損益: \( P_A(t) \)
- HEDGE通常運用損益: \( P_R(t) \)
- 保険ペイオフ: \( P_I(t) \)
- 総費用: \( C(t) \)
- 統合損益: \( P_{\mathrm{total}}(t) = P_A(t) + P_R(t) + P_I(t) - C(t) \)

目標: \( P_{\mathrm{total}}(t) \geq 0 \)（設計目標。成立条件は別途証明・検証）

保護の保証・確率的損失制限・単なる期待収益の改善を混同しないこと。

---

## 2. ProtectiveHedgeの継承（ベースライン）

元の構造を研究上のベースラインとして保存する。変更する場合は元式と改良式を別バージョンで管理。

\[
H_t = \operatorname{clip}\left( \frac{w_t R_t e^{-\delta \Delta t}}{1 + \rho Risk_t}, 0, H_{\max} \right)
\]

\[
w_t = \frac{1}{1 + \exp\left[ -\kappa_1 \left( \frac{\sigma_t}{\sigma_{\mathrm{ref}}} - 1 \right) - \kappa_2 D_t \right]}
\]

\[
D_t = \max\left( \frac{K - S_t}{S_t}, 0 \right)
\]

\[
R_t = \lambda_1 D_t + \lambda_2 \max\left( \frac{\sigma_t - \sigma_{\mathrm{ref}}}{\sigma_{\mathrm{ref}}}, 0 \right) + \lambda_3 \sigma_t \sqrt{\tau}
\]

特に検証する項目:
1. ロング・ショート双方への対応
2. ボラティリティ単位の整合性
3. 時間減衰の妥当性
4. Risk_tによるヘッジ縮小の妥当性
5. ヘッジ強度と必要保護金額の対応
6. 実際のヘッジ数量への変換
7. 保険料・スプレッド・証拠金
8. 保有資産とヘッジ商品の通貨換算
9. 複数資産の共通ショック
10. 保護不足が発生する市場条件

元式が保護を保証していると仮定してはならない。

---

## 3. STEP 1 調査結果（既存資産棚卸し）

### 確認済みリポジトリ（Sai310421）
- **xauusd-data**（public）
  - `research/` に多数のNautilusTrader Raw Tick BTスクリプト（amos_*, ae_*, armada_*, bigplayer_*, minimumspike_raw6x3_bt.py など）
  - `research/build_raw_bidask_catalog_duka.py`: Dukascopy BI5からRaw Bid/Ask QuoteTickカタログ構築
  - `.github/workflows/nautilus-6symbol-3tf-raw-bidask.yml`: Raw BidAsk専用Nautilus実行ワークフロー
  - `research/MULTIEDGE_FINAL_SELECTION.md`: PF≥1.20ゲート、Raw BidAsk必須、コストストレス、安定性検査
- **xauusd-duka-feed**（public）
  - Cache-first XAUUSD Dukascopyフィード。Actionsで欠損日を埋める。
- その他（private含む）: research-line（Nautilus-first）、ML-Trading-Lab、my-ea-factory-ICT、MQL5_Project、mss-amos など

### 照合結果
| 項目 | 状態 |
|------|------|
| NautilusTrader既存実装 | あり（豊富） |
| XAUUSD Bid/Ask Raw Tick | あり |
| BacktestEngine / データカタログ | あり |
| Optuna | 未確認（要追加） |
| OOS/WFO / Monte Carlo | 部分的 |
| Risk Governor | 未確認 |
| 既存RAGE実装 | **なし** |
| ProtectiveHedge | **なし** |

**結論**: Raw Tick検証基盤は再利用可能。ProtectiveHedgeとRAGE Dualは新規実装が必要。

---

## 4. Build実行指示（必須工程）

### STEP A: 数式・会計基盤
- 元のProtectiveHedgeを`protective_hedge.py`で再現。元式と派生式をバージョン管理。
- MAIN / HEDGE独立会計（`compound_accounting.py`, `portfolio_state.py`）。
- 二階層最適化インターフェース（保護能力確保 → 残資本で複利最大化）。

### STEP B: Raw Tickアダプター
- 既存`build_raw_bidask_catalog_duka.py`とNautilusワークフローを拡張。
- XAUUSD Raw Bid/Ask QuoteTick必須。OHLC代理禁止。
- 入力成功条件: `RAW_BIDASK_CATALOG_OK`, QuoteTick ≥ 1000, ask ≥ bid, timestamps monotonic。

### STEP C: エンジン実装
- `main_engine.py` / `hedge_engine.py` / `insurance_pool.py` / `hedge_optimizer.py`。
- 既存`research/`の有望スクリプト（ae_allweather_* , armada_* , bigplayer_* など）をMAIN候補として接続可能なアダプターを用意。

### STEP D: 検証実行
- Nautilus Native Fillで約定確認。
- OOS/WFO（学習・最適化・完全未使用期間を分離。未来情報混入禁止）。
- Monte Carlo（paired block / regime-conditional。MAINとHEDGEの共通ショックを保持。独立シャッフル禁止）。
- 計測項目: 月次収益率（MAIN/HEDGE/統合）、幾何平均成長率、max floating DD、同時損失頻度、保護不足額、破綻確率、コスト控除後純利益。

### STEP E: ゲート判定と報告
- Gate 1〜6を順に通過確認。
- 最終報告形式に従い、ログ・JSON・CSVの証拠を添付。

---

## 5. 二階層最適化（必須）

第一階層: 必要な保護能力を確保  
\[
q_H^* = \arg\min_{q_H} C_H(q_H)
\]
制約: \( P_A(s) + P_H(q_H, s) \geq -B_{\mathrm{loss}} \) （事前定義シナリオ集合 \( \mathcal{S}_{\mathrm{specified}} \) に対して）

第二階層: 保護制約を維持したまま複利成長率を最大化  
\[
\max_{\pi_A, \pi_H} E\left[ \log\left( \frac{W_T}{W_0} \right) \right]
\]
目標: \( r_{H,\mathrm{monthly}} \geq 0.10 \), \( r_{A,\mathrm{monthly}} \geq 0.10 \)

目標未達の場合は最適化問題が実行不能である可能性も含めて報告。制約を黙って緩和し成功扱いにしない。

---

## 6. 必須成果物ディレクトリ

```
rage-dual-compound-hedge/
├── README.md
├── pyproject.toml
├── configs/
│   ├── research.yaml
│   ├── risk_limits.yaml
│   └── instruments.yaml
├── docs/
│   ├── RESEARCH_HYPOTHESIS.md
│   ├── MATH_EDGE_DICTIONARY.md
│   ├── OPEN_SOURCE_COMPARISON.md
│   ├── ARCHITECTURE.md
│   ├── VALIDATION_PROTOCOL.md
│   └── FAILURE_MODES.md
├── src/rage_dch/
│   ├── protective_hedge.py
│   ├── portfolio_state.py
│   ├── main_engine.py
│   ├── hedge_engine.py
│   ├── insurance_pool.py
│   ├── hedge_optimizer.py
│   ├── compound_accounting.py
│   ├── risk_governor_adapter.py
│   └── reporting.py
├── adapters/
│   ├── nautilus/
│   ├── optuna/
│   ├── wfo/
│   └── monte_carlo/
├── tests/ (unit / integration / stress)
├── scripts/
│   ├── validate_raw_ticks.py
│   ├── run_backtest.py
│   ├── run_wfo.py
│   ├── run_monte_carlo.py
│   └── build_report.py
└── .github/workflows/
```

既存リポジトリ（推奨: `Sai310421/xauusd-data`）への統合可。研究成果物は識別可能な独立ディレクトリで管理。

---

## 7. 完了判定ゲート

**Gate 1｜数式**: 元ProtectiveHedge再現、派生式別管理、単位・境界条件・異常値テスト成功  
**Gate 2｜実装**: MAIN/HEDGE独立会計、保護数量計算、費用・資本移動整合性、単体テスト成功  
**Gate 3｜Raw Tick**: 実Raw Bid/Ask読込成功、Nautilus実約定確認、取引ログ保存、KPI JSON出力  
**Gate 4｜OOS/WFO**: 完全未使用期間評価、パラメータ安定性、コスト控除後収益評価  
**Gate 5｜Monte Carlo**: 共通ショック保持、同時損失・保護不足計測、破綻率・DD分布算出  
**Gate 6｜研究仮説**: 各項目を PASS / CONDITIONAL / FAIL / BLOCKED で判定

判定項目:
- MAIN_10_PERCENT_COMPOUND
- HEDGE_10_PERCENT_COMPOUND
- SIMULTANEOUS_LOSS_CONTROL
- PROTECTION_COVERAGE
- PORTFOLIO_POSITIVE_GROWTH
- OOS_REPRODUCIBILITY
- MONTE_CARLO_ROBUSTNESS

---

## 8. Buildに渡すために必要な事項

### 推奨渡し方
1. `Sai310421/xauusd-data` にIssueを作成し、本ファイル全文を貼る（またはリンク）。
2. または同リポジトリに `rage-dual-compound-hedge/` ブランチを切り、本ファイルを配置してPR。

### 実行開始に必要な最小情報
- 対象リポジトリ: `Sai310421/xauusd-data`（推奨）
- 最初の検証期間: 例）2026-07-27から30日（既存duka-feedキャッシュ利用）
- MAIN候補優先順位: 既存 `ae_allweather_*` / `armada_*` / `bigplayer_*` から選択、または新規
- 成果物置き場: リポジトリ内 `rage-dual-compound-hedge/` または指定ディレクトリ
- 報告先: Issueコメント + JSON/CSV/ログ

### 現状で足りているもの
- Raw Tickカタログ構築スクリプト
- Nautilus BacktestEngine実行ワークフロー
- Dukascopyフィード

### 不足しているもの（Buildが新規作成）
- ProtectiveHedge実装
- RAGE Dual独立会計・保護最適化
- 完全なOOS/WFO・paired Monte Carlo接続

### 制約
- 無料GitHub Actions runnerの時間・リソース制限を考慮（最初はsmoke → 小規模）。
- Live Trading OFF厳守。
- 証拠（ログ・JSON・CSV・Commit SHA）のない成功報告禁止。

---

## 9. 最終報告形式（必須項目）

報告には必ず以下を含めること。各数値には取得元ログ・JSON・CSV等の証拠を添付。

```
AMOS RAGE DCH RESEARCH REPORT
1. GitHub調査結果
2. 採用オープンソースと選定理由
3. 再利用したAMOS既存資産
4. 新規実装ファイル
5. 数式・研究仮説の変更点
6. テスト結果
7. Raw Tickデータ件数・期間
8. Nautilus Native Fill件数
9. MAIN月間純収益率
10. HEDGE月間純収益率
11. 統合月間純収益率
12. 最大Floating DD
13. 同時損失率
14. 保護不足額
15. OOS/WFO結果
16. Monte Carlo結果
17. 仮説判定（PASS/CONDITIONAL/FAIL/BLOCKED）
18. Grok独自の改善提案
19. GitHub Issue/PR/Commit SHA
20. 次工程
```

---

## 10. オープンソース暫定選定

- **主エンジン**: NautilusTrader（既存資産と完全互換、Raw Tickネイティブ）
- **最適化**: Optuna（後付け接続）
- **高速スクリーニング**: vectorbtは補助のみ（Raw Tick代替禁止）
- **統計**: arch または既存bootstrap実装優先

より優れた構成が見つかった場合は理由と検証証拠を提示して採用可。

---

## 11. 最初の実行順序（確認待ちで停止しない）

1. 既存資産の再確認（本ファイル第3項をベースに）
2. ProtectiveHedge元式の実装と単体テスト
3. Raw Tickカタログ接続とNautilus smoke実行
4. MAIN/HEDGE独立会計の最小実装
5. 小規模期間でのバックテスト → OOS/WFO → Monte Carlo
6. ゲート判定と証拠付き報告
7. 不成立条件の特定と改良提案

失敗時は原因を特定し代替経路を検討。証拠のない成功報告は禁止。

---

## 最終指令

本プロジェクトは、月利10%を達成したように見せるための研究ではない。  
主力資産とヘッジ資産がそれぞれ複利成長し、互いの損失を補完するRAGE型システムを数学的に成立させ、その再現可能性を実市場データで検証する研究である。

最初から成立仮説を明確に置き、成立条件を探索する。  
ただし、不成立となる条件を隠さず、必要な修正を数学的に導出する。

既存AMOS資産を最大限再利用すること。  
より優れた数学・オープンソース・実装方式がある場合は積極的に提案し、検証可能な形で取り入れること。

**このファイルをそのままBuildに渡して実行開始してください。**
