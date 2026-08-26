# NOTE-HFT Frozen Restore Patch v1

## 原則
- Frozen原本は上書きしない。
- 元ファイルを入力し、別名の派生修正版を出力する。
- 売買ロジックを推測で補完しない。
- 欠落した実行構造だけ修復する。

## 安全に復元する箇所
1. `self.mt_max_pos` の未初期化。
2. Tick / Position / Account の「受信済み」状態管理。
3. 元コード中に完全な形でコメントアウトされている `AccountInfo` snapshot。
4. `wait_for_data()` がゼロポジションを未受信扱いする問題。
5. MT5 slow-order経路欠落時の Fail-Closed。
6. `a_cond_num / b_cond_num` の `00000` 部分を正式な未復元Frozen Fragmentとして明示。

## 変更禁止
`static_qty=0.01`, `c_n_of=0`, 3秒Entry Permit, 10ms loop, Entry/Close条件, BUY/SELL方向, Balance Ratio gate, `max_lev`, Spread gate式, Close-first構造。

## 未復元・推測禁止
- `a_cond_num / b_cond_num`：原型EDGE中心。EMA/ATR/Spike/MLで代理しない。
- `sp_limit_cnt`：供給ソースは0。意図的か欠落か証明できないため変更しない。
- MT5 slow-order transport：元Port/Protocolが無いため勝手に生成しない。

## Parity Gate
- N = 176,483
- WR = 72.71%
- PF = 1.74
- MaxDD = 3.97%
- BUY = 88,223
- SELL = 88,260

復元順序：Structural Parity → Reality Noise → Economic Parity → Supervisor Shadow → Intervention。
