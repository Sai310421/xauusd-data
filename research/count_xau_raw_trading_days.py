from pathlib import Path
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
from datetime import datetime, timezone
import json

cat=ParquetDataCatalog("catalog/raw_bidask")
inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
dates=sorted({datetime.fromtimestamp(int(t.ts_event)/1e9,timezone.utc).date().isoformat() for t in ticks})
weekdays=[d for d in dates if datetime.fromisoformat(d).weekday()<5]
out={"ticks":len(ticks),"first_date":dates[0],"last_date":dates[-1],"unique_dates":len(dates),"weekday_dates":len(weekdays),"dates":dates}
print(json.dumps(out,indent=2))
