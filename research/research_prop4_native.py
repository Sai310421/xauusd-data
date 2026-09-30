"""Raw quote candidate audit. Native fill reconciliation is mandatory.

This is a low-dimensional FP/CVaR stopping experiment, not an HJB/MPC solver.
Fees are an explicit per-round-trip-lot scenario overlay; no cashback is assumed.
"""
import argparse
import datetime as dt
import json
import math
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from tickscalper_prop4_base import *

class PropCfg(Cfg, frozen=True):
 initial_cash: float = 1000.0
 stop_dd: float = 3.5
 commission_rt: float = 0.0
 budget_add: bool = False
 day_timezone: str = 'Europe/Prague'
 final_ts: int = 0

class PropCandidate(TickScalperCandidate):
 def __init__(self, c):
  super().__init__(c)
  self.eq = self.peak = self.mtm_peak = c.initial_cash
  self.cashflow = self.units = self.fees = self.gross_net = 0.0
  self.native_peak = c.initial_cash
  self.native_dd = self.static_loss = self.daily_loss = 0.0
  self.day = None
  self.day_balance = c.initial_cash
  self.halted = False
  self.dd_stops = self.budget_blocks = self.rejections = 0
  self.fill_count = 0
  self.native_baskets = []
  self.basket_cash_start = self.basket_fee_start = 0.0
  self.max_reconcile = 0.0
 def _native_equity(self):
  mark = self.bid if self.units >= 0 else self.ask
  return self.config.initial_cash + self.cashflow + self.units*(mark or 0) - self.fees
 def _measure(self):
  e = self._native_equity()
  self.native_peak = max(self.native_peak, e)
  self.native_dd = max(self.native_dd, 100*(self.native_peak-e)/self.native_peak)
  self.static_loss = max(self.static_loss, 100*(self.config.initial_cash-e)/self.config.initial_cash)
  self.daily_loss = max(self.daily_loss, 100*(self.day_balance-e)/self.config.initial_cash)
 def on_order_filled(self, event):
  q = float(event.last_qty.as_double())
  px = float(event.last_px.as_double())
  signed = q if event.order_side == OrderSide.BUY else -q
  was_flat = abs(self.units)<1e-8
  if was_flat:
   self.basket_cash_start = self.cashflow
   self.basket_fee_start = self.fees
  self.cashflow -= signed*px
  self.units += signed
  self.fees += q/100*self.config.commission_rt/2
  self.fill_count += 1
  if abs(self.units)<1e-8 and not was_flat:
   self.native_baskets.append({'pnl':self.cashflow-self.basket_cash_start-(self.fees-self.basket_fee_start),'ts':int(event.ts_event)})
  self._measure()
 def on_order_rejected(self, event):
  self.rejections += 1
  self.halted = True
 def _floating_pnl(self):
  return super()._floating_pnl()*100 - sum(l for _,l in self.entries)*self.config.commission_rt
 def _close(self, reason):
  if not self.entries: return
  px = self.bid if self.side > 0 else self.ask
  ticket = [(px-p)*self.side*l*100-l*self.config.commission_rt for p,l in self.entries]
  gross = sum((px-p)*self.side*l*100 for p,l in self.entries)
  for _,lot in self.entries: self._submit(-self.side, lot)
  pnl = sum(ticket)
  self.gross_net += gross
  self.ticket_pnls.extend(ticket)
  self.trades.append({'pnl':pnl,'depth':len(self.entries),'reason':reason,'ts':self.now_ns})
  self.net += pnl
  self.eq += pnl
  self.peak = max(self.peak,self.eq)
  self.mdd = max(self.mdd,100*(self.peak-self.eq)/self.peak)
  if pnl > 0: self.gw += pnl
  elif pnl < 0: self.gl -= pnl
  self.side = 0
  self.entries = []
  self.last_close_ns = self.now_ns
  self.last_close_reason = reason
 def _risk_add_scale(self, lot):
  scale = super()._risk_add_scale(lot)
  if not self.config.budget_add: return scale
  gross = sum(l for _,l in self.entries)
  # Conservative one-step add budget; total and daily floors use initial capital.
  floor = max(self.native_peak*(1-self.config.stop_dd/100),
              self.config.initial_cash*(1-self.config.stop_dd/100),
              self.day_balance-self.config.initial_cash*self.config.stop_dd/100)
  available = self._native_equity()-floor
  distance = max(self.config.emergency_distance,self.config.add_distance)
  per_lot = distance*100 + max(0,self.ask-self.bid)*100 + self.config.commission_rt
  permitted = max(0,available/per_lot-gross)
  bounded = min(scale,permitted/max(lot,1e-12))
  if floor_step(lot*bounded) < .01: self.budget_blocks += 1
  return bounded
 def _optimal_stop_values(self):
  z = super()._optimal_stop_values()
  if z is None: return None
  # Correct the stopping boundary: use distance still remaining to Emergency.
  s = self.side
  vals = [d*s for d in self.recent_mid_deltas]
  if len(vals) < 80: return None
  mu = sum(vals)/len(vals)
  var = sum((x-mu)**2 for x in vals)/max(1,len(vals)-1)
  gross = sum(l for _,l in self.entries)
  a = max(0,self.config.emergency_distance-z['adverse_from_last'])
  b = z['upside']/max(gross,1e-12)
  if a <= 0: tail = 1.0
  elif b <= 0: tail = 0.0
  elif var <= 1e-12: tail = 1.0 if mu < 0 else 0.0
  elif abs(mu) < 1e-10: tail = b/(a+b)
  else:
   k = -2*mu/var
   if k*(a+b) > 700:
    up = math.exp(-k*b)*(-math.expm1(-k*a))/(-math.expm1(-k*(a+b)))
   else:
    up = math.expm1(k*a)/math.expm1(k*(a+b))
   tail = max(0,min(1,1-up))
  z['p_tail'] = tail
  z['p_nr'] = 1-tail
  z['ev_continue'] = (1-tail)*z['upside']-tail*z['remaining_tail']-z['wait_cost']
  z['v3_score'] = z['ev_continue']-self.config.optimal_stop_cvar_lambda*z['tail_cvar']
  return z
 def on_quote_tick(self, t):
  self.bid = float(t.bid_price.as_double())
  self.ask = float(t.ask_price.as_double())
  self.now_ns = int(t.ts_event)
  day = dt.datetime.fromtimestamp(t.ts_event/1e9,ZoneInfo(self.config.day_timezone)).date()
  if self.day != day:
   self.day = day
   # Existing floating loss remains counted across the daily reset.
   self.day_balance = self.config.initial_cash+self.gross_net-self.fees
  self._measure()
  e = self._native_equity()
  near = (100*(self.native_peak-e)/self.native_peak >= self.config.stop_dd or
          100*(self.config.initial_cash-e)/self.config.initial_cash >= self.config.stop_dd or
          100*(self.day_balance-e)/self.config.initial_cash >= self.config.stop_dd)
  if near and not self.halted:
   self.halted = True
   self.dd_stops += 1
   self._close('PROP_DD_STOP')
  if t.ts_event >= self.config.final_ts:
   self._close('EOD_QUOTE')
   self.halted = True
   return
  if self.halted: return
  super().on_quote_tick(t)
 def on_stop(self):
  # Pending market orders at engine shutdown must not create shadow profits.
  pass

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument('--catalog',required=True)
 ap.add_argument('--id',required=True)
 ap.add_argument('--initial',type=float,default=1000)
 ap.add_argument('--lot',type=float,default=.01)
 ap.add_argument('--mode',choices=['off','v2','v3'],default='off')
 ap.add_argument('--budget-add',action='store_true')
 ap.add_argument('--fee',type=float,default=0)
 ap.add_argument('--leverage',type=int,default=30)
 a = ap.parse_args()
 if a.lot < .01 or abs(a.lot/.01-round(a.lot/.01)) > 1e-8:
  raise SystemExit('Base lot must match the 0.01 lot minimum and step')
 cat = ParquetDataCatalog(a.catalog)
 inst = next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 ticks = fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
 if not ticks: raise SystemExit('Raw quotes missing')
 ticks.sort(key=lambda t:t.ts_event)
 eng = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(a.initial,USD)],default_leverage=Decimal(a.leverage))
 eng.add_instrument(inst)
 eng.add_data(ticks)
 st = PropCandidate(PropCfg(instrument_id=inst.id,initial_cash=a.initial,base_qty=Decimal(str(a.lot)),stop_dd=3.5,commission_rt=a.fee,budget_add=a.budget_add,final_ts=int(ticks[-1].ts_event),max_layers=9,basket_offset=.8575,emergency_distance=5.75,emergency_cooldown_seconds=55,entry_mode='ticksmoother',ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,risk_mode='ddr_fp_size',ddr_dd_cap=18,ddr_lambda=2.5,ddr_min_scale=.5,ddr_start_layer=8,min_add_scale=.25,fp_threshold=.6,fp_block_threshold=.8,optimal_stop_mode=a.mode,optimal_stop_min_layer=9))
 eng.add_strategy(st)
 eng.run()
 fills = eng.trader.generate_order_fills_report()
 out = Path('results/prop4')/a.id
 out.mkdir(parents=True,exist_ok=True)
 if fills is not None: fills.to_csv(out/'native_fills.csv')
 eng.trader.generate_positions_report().to_csv(out/'native_positions.csv')
 account = eng.cache.account_for_venue(inst.id.venue)
 balance = float(account.balance_total(USD).as_double())
 shadow_error = abs(balance-a.initial-st.gross_net)
 error = abs(balance-a.initial-st.cashflow)
 fill_error = abs(sum(x['pnl'] for x in st.native_baskets)-(st.cashflow-st.fees))
 # USD Money rounds realized P/L to cents on each native execution.
 rounding_bound = .005*st.fill_count+.01
 flat = abs(st.units)<1e-8 and not st.entries
 reconciled = flat and error<=rounding_bound and fill_error<=1e-6 and not st.rejections
 dd_ok = max(st.native_dd,st.static_loss,st.daily_loss)<=4
 native_net = balance-a.initial-st.fees
 native_wins = sum(x['pnl']>0 for x in st.native_baskets)
 native_gw = sum(max(0,x['pnl']) for x in st.native_baskets)
 native_gl = sum(max(0,-x['pnl']) for x in st.native_baskets)
 result = {**st.summary(),'N':len(st.native_baskets),'WR_pct':100*native_wins/max(1,len(st.native_baskets)),'PF':native_gw/native_gl if native_gl else None,'EV':native_net/max(1,len(st.native_baskets)),'Net':native_net,'Return_pct':100*native_net/a.initial,'initial_cash':a.initial,'base_lot':a.lot,'leverage':a.leverage,'commission_rt_overlay':a.fee,'native_net_after_fee':balance-a.initial-st.fees,'native_return_pct':100*(balance-a.initial-st.fees)/a.initial,'native_peak_DD_pct':st.native_dd,'initial_capital_loss_pct':st.static_loss,'daily_loss_pct':st.daily_loss,'daily_reset_timezone':st.config.day_timezone,'native_balance':balance,'native_reconcile_error_usd':error,'usd_rounding_bound':rounding_bound,'shadow_reconcile_error_usd':shadow_error,'kpi_source':'NATIVE_FILL_CASHFLOW_AND_ACCOUNT_BALANCE','fill_cashflow_error_usd':fill_error,'native_flat':flat,'native_fill_count':st.fill_count,'rejections':st.rejections,'DD4_pass':bool(reconciled and dd_ok),'accounting_pass':bool(reconciled),'dd_stops':st.dd_stops,'budget_blocks':st.budget_blocks,'budget_add':a.budget_add,'raw_ticks':len(ticks),'start_ns':int(ticks[0].ts_event),'end_ns':int(ticks[-1].ts_event),'status':'RESEARCH_VALID' if reconciled else 'INVALID_ACCOUNTING','limitations':['Deterministic immediate fills; no latency/slippage stress','FP/CVaR heuristic stopping; not a solved HJB or MPC','Commission is an overlay scenario, not a verified prop fee schedule','Native margin model uses the catalog CurrencyPair instrument'],'ohlc_resample_used':False}
 (out/'kpi.json').write_text(json.dumps(result,indent=2))
 (out/'baskets.json').write_text(json.dumps({'native_baskets':st.native_baskets,'shadow_decisions':st.trades},indent=2))
 print(json.dumps(result,indent=2))
 eng.dispose()
 if not reconciled: raise SystemExit('Native accounting reconciliation failed')

if __name__ == '__main__': main()
