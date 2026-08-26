#!/usr/bin/env python3
from pathlib import Path
import sys, hashlib, json

def sha256_text(s:str)->str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def replace_once(src, old, new, label):
    n=src.count(old)
    if n!=1:
        raise RuntimeError(f'{label}: expected exactly 1 match, found {n}; unsafe patch refused')
    return src.replace(old,new,1)

def main():
    if len(sys.argv)!=3:
        print('usage: python NOTE_HFT_apply_restore_patch_v1.py ORIGINAL.py REPAIRED.py')
        raise SystemExit(2)
    src_path=Path(sys.argv[1]); dst_path=Path(sys.argv[2])
    src=src_path.read_text(encoding='utf-8')
    before=sha256_text(src); changes=[]

    old="""        self.max_pos=0\n        self.mt_max_pos\n        self.set_odr_qty=0"""
    new="""        self.max_pos=0\n        self.mt_max_pos=Decimal('0')  # RESTORE: existing attribute initialization only\n        self.set_odr_qty=0"""
    if old in src:
        src=replace_once(src,old,new,'mt_max_pos init'); changes.append('mt_max_pos initialization restored')

    old="""        self.latest_t: Optional[MarketData] = None\n        self.latest_positions: list[Position] = []\n        self.latest_account: Optional[AccountInfo] = None"""
    new="""        self.latest_t: Optional[MarketData] = None\n        self.latest_positions: list[Position] = []\n        self.latest_account: Optional[AccountInfo] = None\n        self._got_tick = False\n        self._got_positions = False\n        self._got_account = False"""
    if old in src:
        src=replace_once(src,old,new,'receive flags'); changes.append('receive state flags restored')

    old="""                    # pos\n                    if len([x for x in self.latest_positions if x.symbol==sym]):"""
    new="""                    self._got_positions = True  # empty snapshot is valid\n\n                    # pos\n                    if len([x for x in self.latest_positions if x.symbol==sym]):"""
    if old in src:
        src=replace_once(src,old,new,'position flag'); changes.append('position snapshot flag restored')

    old="""                    #self.latest_account = AccountInfo(\n                    #    balance=float(data.get('balance')),\n                    #    equity=float(data.get('equity')),\n                    #    margin=float(data.get('margin')),\n                    #    margin_free=float(data.get('margin_free')),\n                    #    margin_level=float(data.get('margin_level')),\n                    #    profit=float(data.get('profit')),\n                    #    timestamp=int(data.get('timestamp'))\n                    #)"""
    new="""                    self.latest_account = AccountInfo(\n                        balance=float(data.get('balance')),\n                        equity=float(data.get('equity')),\n                        margin=float(data.get('margin')),\n                        margin_free=float(data.get('margin_free')),\n                        margin_level=float(data.get('margin_level')),\n                        profit=float(data.get('profit')),\n                        timestamp=int(data.get('timestamp'))\n                    )\n                    self._got_account = True"""
    if old in src:
        src=replace_once(src,old,new,'AccountInfo restore'); changes.append('AccountInfo snapshot restored from commented original structure')

    old="""            if self.latest_t and self.latest_positions and self.latest_account:\n                logger.info(\"[INFO] All data received\")\n                return True"""
    new="""            if self._got_tick and self._got_positions and self._got_account:\n                logger.info(\"[INFO] All data received\")\n                return True"""
    if old in src:
        src=replace_once(src,old,new,'wait_for_data'); changes.append('wait_for_data zero-position deadlock fixed')

    old="""            else:\n                await self.sockets['order_slow'].send_string(json.dumps(order_data))\n                logger.info(f\"[ORDER_SLOW] {cmd} {symbol} {volume}x (SL:{sl}, TP:{tp})\")\n                self.MTODR_PERMIT=False"""
    new="""            else:\n                if 'order_slow' not in self.sockets:\n                    raise RuntimeError('MT5 slow-order transport is absent from the supplied Frozen source; attach the original MT5 adapter. No port/protocol was guessed.')\n                await self.sockets['order_slow'].send_string(json.dumps(order_data))\n                logger.info(f\"[ORDER_SLOW] {cmd} {symbol} {volume}x (SL:{sl}, TP:{tp})\")\n                self.MTODR_PERMIT=False"""
    if old in src:
        src=replace_once(src,old,new,'slow order guard'); changes.append('missing MT5 slow-order route made fail-closed')

    old="""                if 00000 and 00000:\n                    self.a_cond_num=00000\n                    self.b_cond_num=00000\n                else:\n                    self.a_cond_num=0\n                    self.b_cond_num=0"""
    new="""                # FROZEN MISSING FRAGMENT: original a_cond_num / b_cond_num generator is masked.\n                # DO NOT replace this with EMA/ATR/Spike/proxy logic.\n                if 00000 and 00000:\n                    self.a_cond_num=00000\n                    self.b_cond_num=00000\n                else:\n                    self.a_cond_num=0\n                    self.b_cond_num=0"""
    if old in src:
        src=replace_once(src,old,new,'masked signal marker'); changes.append('masked alpha fragment explicitly protected; behavior unchanged')

    dst_path.write_text(src,encoding='utf-8')
    report={'source':str(src_path),'output':str(dst_path),'source_sha256':before,'output_sha256':sha256_text(src),'changes':changes,'unresolved_do_not_guess':['a_cond_num / b_cond_num original generator','sp_limit_cnt original intent/value if redacted','original MT5 slow-order transport'],'frozen_untouched':['static_qty=0.01','c_n_of=0','3-second create permit','10ms loop sleep','entry/close conditions','BUY/SELL mapping','balance-ratio gate','max_lev','spread gate expression','close-first structure']}
    dst_path.with_suffix(dst_path.suffix+'.restore_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
