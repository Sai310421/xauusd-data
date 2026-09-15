from __future__ import annotations

import csv
import html
import re
from datetime import datetime, timezone
from pathlib import Path


def _num(v: str) -> float | None:
    if v is None: return None
    s=str(v).strip().replace('\u00a0',' ').replace(',','')
    m=re.search(r'[-+]?\d+(?:\.\d+)?',s)
    return float(m.group()) if m else None


def _ts(v: str) -> float | None:
    s=str(v).strip()
    for fmt in ('%Y.%m.%d %H:%M:%S','%Y.%m.%d %H:%M','%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M'):
        try: return datetime.strptime(s,fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError: pass
    n=_num(s)
    return n


def _strip_html(text: str) -> list[list[str]]:
    rows=[]
    for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',text,flags=re.I|re.S):
        cells=[]
        for td in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,flags=re.I|re.S):
            td=re.sub(r'<br\s*/?>',' ',td,flags=re.I)
            td=re.sub(r'<[^>]+>','',td)
            cells.append(html.unescape(td).strip())
        if cells: rows.append(cells)
    return rows


def import_mt5_html(path: str) -> list[dict]:
    """Best-effort importer for exported MT5 Strategy Tester HTML.

    It extracts observable closed deal rows. MT5 report layouts vary by build/language,
    so unrecognized rows are skipped rather than guessed.
    """
    text=Path(path).read_text(encoding='utf-8',errors='ignore')
    if '<html' not in text.lower():
        text=Path(path).read_text(encoding='utf-16',errors='ignore')
    rows=_strip_html(text)
    deals=[]
    # Common Deals table contains time/deal/symbol/type/direction/volume/price/order/commission/swap/profit/balance/comment.
    for c in rows:
        low=[x.lower() for x in c]
        if len(c) < 7: continue
        if not any(x in ('buy','sell') for x in low): continue
        side='buy' if 'buy' in low else 'sell'
        time_i=next((i for i,x in enumerate(c) if _ts(x) is not None and ('.' in x or '-' in x)),None)
        if time_i is None: continue
        nums=[(i,_num(x)) for i,x in enumerate(c)]
        nums=[x for x in nums if x[1] is not None]
        if not nums: continue
        # Profit is normally near the right side. Preserve raw row for audit.
        profit=nums[-2][1] if len(nums)>=2 else nums[-1][1]
        price=next((v for i,v in nums if i>time_i and v and v>0),None)
        deals.append({'deal_ts':_ts(c[time_i]),'side':side,'price':price,'pnl_component':profit,'raw':' | '.join(c)})
    return deals


def pair_deals_to_trades(deals: list[dict]) -> list[dict]:
    """Conservative observable pairing.

    Pairs opposite-side deal events sequentially. Complex partial closes/grid baskets
    must be handled by a later position/deal-id aware adapter; no hidden behavior is inferred.
    """
    out=[]; open_deal=None
    for d in deals:
        if open_deal is None:
            open_deal=d; continue
        if d['side']==open_deal['side']:
            continue
        out.append({'entry_ts':open_deal['deal_ts'],'exit_ts':d['deal_ts'],'side':open_deal['side'],
                    'entry':open_deal['price'],'exit':d['price'],'pnl':d.get('pnl_component',0.0),
                    'add_count':0,'source_id':'mt5_tester_html'})
        open_deal=None
    return out


def write_normalized(rows: list[dict], out: str):
    fields=['entry_ts','exit_ts','side','entry','exit','pnl','add_count','source_id']
    with open(out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
