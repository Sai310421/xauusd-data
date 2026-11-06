# TickScalper G1 — MT5初期バックテスト設定

対象EA: `TickScalper_G1_RiskGoverned.mq5`  
設定ファイル: `TickScalper_G1_BT_Initial.set`

## ストラテジーテスター設定

| 項目 | 初期値 |
|---|---|
| EA | `TickScalper_G1_RiskGoverned` |
| 銘柄 | ブローカーのXAUUSD銘柄を選択（例: XAUUSD / XAUUSDm） |
| 期間 | M1 |
| モード | 実ティックベース（Every tick based on real ticks） |
| 口座 | Hedging口座 |
| レバレッジ | 1:2000（実運用予定と同条件） |
| 初期証拠金 | $1,000（比較の基準値） |
| 手数料・スプレッド | ブローカー実値。固定ゼロは不可 |
| 最適化 | 初回はOFF |

`TradeSymbol` は空欄です。テスターで選んだ銘柄をそのまま使うため、銘柄サフィックス差異で注文が出ないことを避けます。

## 目的

このプリセットは利益最大化用ではありません。まず以下を確認するための、安全側の初期値です。

1. M1実ティック上でBuy/Sellの注文が実際に発生すること
2. 個別SL／TP、最大2追加、DD停止、金曜停止が機能すること
3. 追加ロットが1.5倍・十段型に戻らないこと
4. `EnableTrading=true` がテスターでのみ読み込まれていること

## 初回の合否

- 取引数が0: 銘柄、実ティック、EA有効化、ジャーナルの拒否理由を先に確認する
- 取引数が1以上: 約定、SL、TP、Magic、最大追加数、停止ログを保存する
- 採用判定: MT5結果だけでは行わない。Nautilus Raw Bid/AskのOOS・コスト感度・Monte Carloと照合する

この設定はDEMO／本番の推奨値ではありません。
