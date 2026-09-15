from __future__ import annotations
import argparse, json
from .importer import import_mt5_html, pair_deals_to_trades, write_normalized


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--html',required=True)
    p.add_argument('--out',default='normalized_trades.csv')
    a=p.parse_args()
    deals=import_mt5_html(a.html)
    trades=pair_deals_to_trades(deals)
    write_normalized(trades,a.out)
    print(json.dumps({'observable_deals':len(deals),'normalized_trades':len(trades),'out':a.out},indent=2))

if __name__=='__main__': main()
