from __future__ import annotations
import argparse, json, math, hashlib, os, datetime
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

TF_MIN=(1,5,15)
INITIAL=1000.0
LEVERAGE=2000.0
BASE_QTY=1.0       # 1 oz ~= 0.01 lot if XAUUSD contract is 100 oz/lot
BOOST_QTY=3.0      # ~= 0.03 lot
FIBS=(0.786,0.804,0.822)
BOOST_WINDOW_NS=45*60*1_000_000_000
TIMEOUT_NS=240*60*1_000_000_000

def ff(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

@dataclass
class BarState:
    start_ns:int=0; o:float=0; h:float=0; l:float=0; c:float=0
    hist:list=field(default_factory=list)

@dataclass
class Wave:
    state:str='IDLE'
    direction:int=0
    origin:float=0.0
    end:float=0.0
    atr:float=0.0
    setup_ns:int=0
    touched:bool=False
    active:bool=False
    entry_ns:int=0
    entry:float=0.0
    sl:float=0.0
    tp:float=0.0

@dataclass
class Leg:
    kind:str
    tf:int
    direction:int
    qty:float
    entry:float
    entry_ns:int
    sl:float|None=None
    tp:float|None=None
    active:bool=True

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    initial_balance: float=INITIAL

class ElliottRawMTF(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.bars={m:BarState() for m in TF_MIN}
        self.waves={m:Wave() for m in TF_MIN}
        self.legs=[]
        self.closed=[]
        self.last_bid=None; self.last_ask=None; self.last_ns=0
        self.realized=0.0; self.peak_equity=INITIAL; self.max_float_dd=0.0
        self.min_margin_level=math.inf; self.max_gross_qty=0.0
        self.triple_latched=False; self.submitted=0
        self.first_ns=None; self.last_seen_ns=None

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _submit(self,side,qty):
        inst=self.cache.instrument(self.config.instrument_id)
        q=inst.make_qty(Decimal(str(qty)))
        o=self.order_factory.market(instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=q)
        self.submit_order(o); self.submitted+=1

    def _bar_update(self,m,ts,px):
        span=m*60*1_000_000_000
        bucket=(ts//span)*span
        b=self.bars[m]
        closed=None
        if b.start_ns==0:
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px
        elif bucket==b.start_ns:
            b.h=max(b.h,px); b.l=min(b.l,px); b.c=px
        else:
            closed=(b.start_ns,b.o,b.h,b.l,b.c)
            b.hist.append(closed)
            if len(b.hist)>256: b.hist=b.hist[-256:]
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px
        return closed

    def _atr14(self,hist):
        if len(hist)<16:return 0.0
        arr=hist[-16:]
        trs=[]
        for i in range(1,len(arr)):
            _,o,h,l,c=arr[i]; pc=arr[i-1][4]
            trs.append(max(h-l,abs(h-pc),abs(l-pc)))
        return float(np.mean(trs[-14:]))

    def _detect(self,m):
        hist=self.bars[m].hist
        if len(hist)<35:return
        w=self.waves[m]
        if w.state!='IDLE':return
        cur=hist[-1]; ts,o,h,l,c=cur
        prev20=hist[-21:-1]
        avg_body=np.mean([abs(x[4]-x[1]) for x in prev20])
        atr=self._atr14(hist)
        if avg_body<=0 or atr<=0:return
        body=abs(c-o); rng=h-l
        prior3=hist[-4:-1]
        ph=max(x[2] for x in prior3); pl=min(x[3] for x in prior3)
        bull_disp=c>o and body>=1.6*avg_body and rng>=0.9*atr and c>ph
        bear_disp=c<o and body>=1.6*avg_body and rng>=0.9*atr and c<pl
        bull_origin=None; bear_origin=None
        # search 6 preceding bars for sweep of older 12-bar range
        for off in range(2,8):
            idx=len(hist)-off
            if idx<12: continue
            bar=hist[idx]; older=hist[idx-12:idx]
            ol=min(x[3] for x in older); oh=max(x[2] for x in older)
            if bull_origin is None and bar[3]<ol and bar[4]>ol: bull_origin=bar[3]
            if bear_origin is None and bar[2]>oh and bar[4]<oh: bear_origin=bar[2]
        if bull_disp and bull_origin is not None and h>bull_origin:
            self.waves[m]=Wave('ARMED',1,bull_origin,h,atr,ts)
        elif bear_disp and bear_origin is not None and bear_origin>l:
            self.waves[m]=Wave('ARMED',-1,bear_origin,l,atr,ts)

    def _mss_after_touch(self,m):
        hist=self.bars[m].hist
        if len(hist)<2:return False
        w=self.waves[m]
        a,b=hist[-1],hist[-2]
        return (a[4]>b[2]) if w.direction>0 else (a[4]<b[3])

    def _open_wave(self,m,bid,ask,ts):
        w=self.waves[m]
        px=ask if w.direction>0 else bid
        span=abs(w.end-w.origin)
        sl=w.origin-w.direction*0.2*w.atr
        tp=w.end+w.direction*0.618*span
        self._submit(w.direction,BASE_QTY)
        self.legs.append(Leg('BASE',m,w.direction,BASE_QTY,px,ts,sl,tp,True))
        w.active=True; w.state='ACTIVE'; w.entry_ns=ts; w.entry=px; w.sl=sl; w.tp=tp

    def _process_wave_tick(self,m,bid,ask,ts):
        w=self.waves[m]
        if w.state=='IDLE':return
        if ts-w.setup_ns>TIMEOUT_NS and not w.active:
            self.waves[m]=Wave();return
        invalid=(bid < w.origin-0.2*w.atr) if w.direction>0 else (ask > w.origin+0.2*w.atr)
        if invalid and not w.active:
            self.waves[m]=Wave();return
        if w.state=='ARMED' and not w.touched:
            # deep retracement band: first touch of any 78.6/80.4/82.2 level
            levels=[w.end-w.direction*abs(w.end-w.origin)*f for f in FIBS]
            if w.direction>0:
                if bid <= max(levels): w.touched=True
            else:
                if ask >= min(levels): w.touched=True

    def _after_bar_close(self,m,bid,ask,ts):
        self._detect(m)
        w=self.waves[m]
        if w.state=='ARMED' and w.touched and self._mss_after_touch(m):
            self._open_wave(m,bid,ask,ts)

    def _close_leg(self,leg,bid,ask,ts,reason):
        if not leg.active:return
        px=bid if leg.direction>0 else ask
        pnl=(px-leg.entry)*leg.direction*leg.qty
        self._submit(-leg.direction,leg.qty)
        leg.active=False; self.realized+=pnl
        self.closed.append({'kind':leg.kind,'tf':leg.tf,'direction':leg.direction,'qty':leg.qty,
            'entry':leg.entry,'exit':px,'pnl':pnl,'entry_ns':leg.entry_ns,'exit_ns':ts,'reason':reason})
        if leg.kind=='BASE':
            self.waves[leg.tf]=Wave()

    def _manage_exits(self,bid,ask,ts):
        for leg in list(self.legs):
            if not leg.active:continue
            mark=bid if leg.direction>0 else ask
            if leg.kind=='BASE':
                hit_sl = mark<=leg.sl if leg.direction>0 else mark>=leg.sl
                hit_tp = mark>=leg.tp if leg.direction>0 else mark<=leg.tp
                if hit_sl:self._close_leg(leg,bid,ask,ts,'SL')
                elif hit_tp:self._close_leg(leg,bid,ask,ts,'TP')
        # boost exits when triple base confluence breaks
        active_base=[x for x in self.legs if x.active and x.kind=='BASE']
        tfdirs={x.tf:x.direction for x in active_base}
        triple=len(tfdirs)==3 and len(set(tfdirs.values()))==1
        if not triple:
            for leg in list(self.legs):
                if leg.active and leg.kind=='BOOST': self._close_leg(leg,bid,ask,ts,'TRIPLE_BREAK')
            self.triple_latched=False

    def _try_boost(self,bid,ask,ts):
        active=[x for x in self.legs if x.active and x.kind=='BASE']
        bytf={}
        for x in active: bytf[x.tf]=x
        if len(bytf)!=3:return
        dirs={x.direction for x in bytf.values()}
        if len(dirs)!=1:return
        times=[x.entry_ns for x in bytf.values()]
        if max(times)-min(times)>BOOST_WINDOW_NS:return
        if self.triple_latched:return
        d=next(iter(dirs)); px=ask if d>0 else bid
        self._submit(d,BOOST_QTY)
        self.legs.append(Leg('BOOST',0,d,BOOST_QTY,px,ts,None,None,True))
        self.triple_latched=True

    def _equity_risk(self,bid,ask):
        floating=0.0; gross=0.0
        for x in self.legs:
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            floating+=(mark-x.entry)*x.direction*x.qty
            gross+=x.qty
        eq=INITIAL+self.realized+floating
        self.peak_equity=max(self.peak_equity,eq)
        self.max_float_dd=max(self.max_float_dd,self.peak_equity-eq)
        self.max_gross_qty=max(self.max_gross_qty,gross)
        mid=(bid+ask)/2
        margin=(gross*mid/LEVERAGE) if gross>0 else 0.0
        if margin>0:self.min_margin_level=min(self.min_margin_level,eq/margin*100.0)

    def on_quote_tick(self,t:QuoteTick):
        bid,ask=ff(t.bid_price),ff(t.ask_price); ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts
        mid=(bid+ask)/2
        closed=[]
        for m in TF_MIN:
            c=self._bar_update(m,ts,bid) # BID bars from raw quotes for signal logic
            if c is not None: closed.append(m)
            self._process_wave_tick(m,bid,ask,ts)
        for m in closed:self._after_bar_close(m,bid,ask,ts)
        self._manage_exits(bid,ask,ts)
        self._try_boost(bid,ask,ts)
        self._equity_risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.legs):
            if x.active:self._close_leg(x,self.last_bid,self.last_ask,self.last_ns,'EOD')

    def result(self):
        p=np.array([x['pnl'] for x in self.closed],float)
        gp=p[p>0].sum() if len(p) else 0.0; gl=-p[p<0].sum() if len(p) else 0.0
        pf=gp/gl if gl>0 else (math.inf if gp>0 else 0.0)
        wr=float((p>0).mean()*100) if len(p) else 0.0
        rf=float(p.sum()/self.max_float_dd) if self.max_float_dd>0 else None
        start=pd.Timestamp(self.first_ns,unit='ns',tz='UTC').date(); end=pd.Timestamp(self.last_seen_ns,unit='ns',tz='UTC').date()
        bdays=max(1,len(pd.bdate_range(start,end)))
        scale=21.0/bdays; net21=float(p.sum()*scale); monthly21=net21/INITIAL*100.0
        daily=((1+monthly21/100.0)**(1/21)-1)*100 if 1+monthly21/100.0>0 else None
        return {'N':int(len(p)),'WR_pct':wr,'PF':float(pf),'RF':rf,'Net_USD':float(p.sum()),
            'Return_pct':float(p.sum()/INITIAL*100),'MaxFloatingDD_USD':float(self.max_float_dd),
            'MaxFloatingDD_pct_initial':float(self.max_float_dd/INITIAL*100),
            'MinMarginLevel_pct_approx':None if math.isinf(self.min_margin_level) else float(self.min_margin_level),
            'MaxGrossQty_oz':float(self.max_gross_qty),'MaxGrossLots_approx':float(self.max_gross_qty/100.0),
            'BusinessDays':bdays,'Monthly21_pct_linearized':float(monthly21),'Net21_USD_linearized':net21,
            'Daily_pct_compound_from_Monthly21':daily,'N21_linearized':float(len(p)*scale),
            'submitted_orders':self.submitted,'base_closed':sum(x['kind']=='BASE' for x in self.closed),
            'boost_closed':sum(x['kind']=='BOOST' for x in self.closed)}

def executable(xs):
    one=Quantity.from_int(1); out=[]; repl=0
    for t in xs:
        bs=t.bid_size; a=t.ask_size
        bsv=ff(bs); av=ff(a)
        if bsv<=0 or av<=0:
            out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if bsv<=0 else bs,ask_size=one if av<=0 else a,ts_event=t.ts_event,ts_init=t.ts_init)); repl+=1
        else: out.append(t)
    return out,repl


def broker_reality_overlay(ticks, trades, initial=INITIAL, leverage=LEVERAGE):
    scenarios = [
        {'name':'RAW_0MS','latency_ms':0,'slip_spread_mult':0.0,'commission_rt_per_lot':0.0,'cashback_rt_per_lot':0.0},
        {'name':'MARKET_NEAR_100MS_C7','latency_ms':100,'slip_spread_mult':0.0,'commission_rt_per_lot':7.0,'cashback_rt_per_lot':0.0},
        {'name':'MARKET_NEAR_100MS_C7_CB6','latency_ms':100,'slip_spread_mult':0.0,'commission_rt_per_lot':7.0,'cashback_rt_per_lot':6.0},
        {'name':'STRESS_300MS_SPREAD10_C7','latency_ms':300,'slip_spread_mult':0.10,'commission_rt_per_lot':7.0,'cashback_rt_per_lot':0.0},
    ]
    outputs={}
    if not ticks:
        return outputs

    first_ns=int(ticks[0].ts_event); last_ns=int(ticks[-1].ts_event)
    start=pd.Timestamp(first_ns,unit='ns',tz='UTC').date()
    end=pd.Timestamp(last_ns,unit='ns',tz='UTC').date()
    bdays=max(1,len(pd.bdate_range(start,end)))

    for sc in scenarios:
        lat_ns=int(sc['latency_ms']*1_000_000)
        events=[]
        for i,tr in enumerate(trades):
            events.append((int(tr['entry_ns'])+lat_ns,0,i,int(tr['entry_ns'])))
            events.append((int(tr['exit_ns'])+lat_ns,1,i,int(tr['exit_ns'])))
        events.sort(key=lambda x:(x[0],x[1]))
        eidx=0
        states={}
        trade_pnl=[]
        fill_delays=[]
        realized=0.0
        long_qty=0.0; long_basis=0.0
        short_qty=0.0; short_basis=0.0
        peak=initial; maxdd=0.0; min_ml=math.inf; max_gross=0.0
        spread_sample=[]

        for k,t in enumerate(ticks):
            ts=int(t.ts_event); bid=ff(t.bid_price); ask=ff(t.ask_price); spread=max(0.0,ask-bid)
            while eidx < len(events) and events[eidx][0] <= ts:
                target,etype,i,signal_ns=events[eidx]
                tr=trades[i]; d=int(tr['direction']); qty=float(tr['qty']); lots=qty/100.0
                slip=spread*float(sc['slip_spread_mult'])
                half_comm=lots*float(sc['commission_rt_per_lot'])/2.0
                if etype==0:
                    px=(ask+slip) if d>0 else (bid-slip)
                    realized -= half_comm
                    states[i]={'entry':px,'direction':d,'qty':qty,'active':True}
                    if d>0:
                        long_qty += qty; long_basis += px*qty
                    else:
                        short_qty += qty; short_basis += px*qty
                else:
                    st=states.get(i)
                    if st and st['active']:
                        px=(bid-slip) if d>0 else (ask+slip)
                        gross=(px-st['entry'])*d*qty
                        cashback=lots*float(sc['cashback_rt_per_lot'])
                        realized += gross-half_comm+cashback
                        net_trade=gross-(2.0*half_comm)+cashback
                        trade_pnl.append(net_trade)
                        if d>0:
                            long_qty -= qty; long_basis -= st['entry']*qty
                        else:
                            short_qty -= qty; short_basis -= st['entry']*qty
                        st['active']=False
                fill_delays.append((ts-signal_ns)/1_000_000.0)
                eidx+=1

            floating=(bid*long_qty-long_basis)+(short_basis-ask*short_qty)
            equity=initial+realized+floating
            peak=max(peak,equity); maxdd=max(maxdd,peak-equity)
            gross_qty=long_qty+short_qty; max_gross=max(max_gross,gross_qty)
            if gross_qty>0:
                margin=gross_qty*((bid+ask)/2.0)/leverage
                if margin>0:min_ml=min(min_ml,equity/margin*100.0)
            if k % 1000 == 0:
                spread_sample.append(spread)

        p=np.array(trade_pnl,float)
        gp=float(p[p>0].sum()) if len(p) else 0.0
        gl=float(-p[p<0].sum()) if len(p) else 0.0
        pf=(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0)
        net=float(p.sum()) if len(p) else 0.0
        scale=21.0/bdays
        monthly21=(net*scale)/initial*100.0
        delays=np.array(fill_delays,float) if fill_delays else np.array([],float)
        spr=np.array(spread_sample,float) if spread_sample else np.array([],float)
        outputs[sc['name']]={
            **sc,'N':int(len(p)),'WR_pct':float((p>0).mean()*100.0) if len(p) else 0.0,
            'PF':float(pf),'Net_USD':net,'Return_pct':net/initial*100.0,
            'MaxFloatingDD_USD':float(maxdd),'MaxFloatingDD_pct_initial':float(maxdd/initial*100.0),
            'RF':float(net/maxdd) if maxdd>0 else None,
            'MinMarginLevel_pct_approx':None if math.isinf(min_ml) else float(min_ml),
            'MaxGrossQty_oz':float(max_gross),'MaxGrossLots_approx':float(max_gross/100.0),
            'BusinessDays':bdays,'Monthly21_pct_linearized':float(monthly21),
            'Net21_USD_linearized':float(net*scale),'N21_linearized':float(len(p)*scale),
            'FillDelay_ms_median':float(np.median(delays)) if len(delays) else None,
            'FillDelay_ms_p95':float(np.percentile(delays,95)) if len(delays) else None,
            'Spread_abs_median_sampled':float(np.median(spr)) if len(spr) else None,
            'Spread_abs_p95_sampled':float(np.percentile(spr,95)) if len(spr) else None,
            'note':'Signals are frozen from the canonical Raw Bid/Ask Nautilus run; execution is replayed on the first raw quote at/after the latency target. Commission/cashback are explicit scenario assumptions.'
        }
    return outputs

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',action='append',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--dataset-id',default='xauusd-raw-bidask'); ap.add_argument('--broker-reality',action='store_true')
    a=ap.parse_args()
    catalogs=[ParquetDataCatalog(p) for p in a.catalog]
    inst=next((x for x in catalogs[0].instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    raw=[]
    for cat in catalogs:
        ci=next((x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
        if ci is None:raise SystemExit('XAUUSD missing in one catalog')
        raw.extend(cat.query(data_cls=QuoteTick,identifiers=[ci.id.value]))
    if not raw:raise SystemExit('no raw XAUUSD QuoteTicks')
    raw.sort(key=lambda t:int(t.ts_event))
    ticks,repl=executable(raw)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
        base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    st=ElliottRawMTF(Cfg(instrument_id=inst.id)); eng.add_strategy(st); eng.run(); eng.end()
    fills=eng.trader.generate_order_fills_report()
    outdir=Path('results/elliott-ict-mtf-raw')/a.experiment_id; outdir.mkdir(parents=True,exist_ok=True)
    trades=pd.DataFrame(st.closed); trades.to_csv(outdir/'trades.csv',index=False)
    result={'verification_level':'NAUTILUS_BT','engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'experiment_id':a.experiment_id,
        'raw_ticks':len(raw),'execution_ticks':len(ticks),'ohlc_resample_used':False,
        'signal_bars':'M1/M5/M15 constructed online from raw BID quotes; fills/exits driven by raw Bid/Ask QuoteTicks',
        'initial_usd':INITIAL,'leverage':LEVERAGE,'base_qty_oz':BASE_QTY,'boost_qty_oz':BOOST_QTY,
        'fib_levels':FIBS,'raw_zero_size_quotes_replaced':repl,
        'native_fills':int(len(fills)) if fills is not None else 0,
        'execution_note':'Nonpositive L1 sizes replaced with Quantity(1) only to permit native matching; raw prices/timestamps unchanged.',
        'margin_note':'MinMarginLevel is an approximate gross-notional/leverage diagnostic, not broker-specific hedged-margin accounting.',
        **st.result()}
    def subset_metrics(rows):
        p=np.array([x['pnl'] for x in rows],float); gp=p[p>0].sum() if len(p) else 0.0; gl=-p[p<0].sum() if len(p) else 0.0
        return {'N':int(len(p)),'WR_pct':float((p>0).mean()*100) if len(p) else 0.0,'PF':float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0),'Net_USD':float(p.sum())}
    result['per_tf']={str(tf):subset_metrics([x for x in st.closed if x['kind']=='BASE' and x['tf']==tf]) for tf in TF_MIN}
    result['boost_only']=subset_metrics([x for x in st.closed if x['kind']=='BOOST'])
    if a.broker_reality:
        result['broker_reality']=broker_reality_overlay(ticks, st.closed)
        (outdir/'broker_reality.json').write_text(json.dumps(result['broker_reality'],indent=2,default=str),encoding='utf-8')
    (outdir/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    manifest={'experiment_id':a.experiment_id,'verification_level':'NAUTILUS_BT','git_sha':os.getenv('GITHUB_SHA'),
        'github_run_id':os.getenv('GITHUB_RUN_ID'),'workflow':os.getenv('GITHUB_WORKFLOW'),'nautilus_version':'1.230.0',
        'dataset_id':a.dataset_id,'strategy_sha256':sha(__file__),
        'config_sha256':hashlib.sha256(json.dumps({'initial':INITIAL,'leverage':LEVERAGE,'fibs':FIBS,'boost_window_min':45,'base_qty':BASE_QTY,'boost_qty':BOOST_QTY},sort_keys=True).encode()).hexdigest(),
        'started_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str)); eng.dispose()

if __name__=='__main__':main()
