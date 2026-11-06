//+------------------------------------------------------------------+
//| TickScalper_G1_RiskGoverned.mq5                                 |
//| Clean-room research EA: risk-governed transformation.           |
//| It does NOT reproduce any third-party source code.              |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property description "Research-only G1 capped recovery controller for XAUUSD MT5 hedging accounts."

#include <Trade/Trade.mqh>

input bool   EnableTrading                    = false; // Must be explicitly enabled after DEMO validation.
input string TradeSymbol                      = "";    // Empty = chart symbol.
input ulong  BuyMagic                         = 260701;
input ulong  SellMagic                        = 260702;
input double BaseLots                         = 0.01;
input int    MaxAdds                          = 2;
input double MaxTotalRiskR                    = 2.00;
input double HardStopATR                      = 1.00;
input double TargetR                          = 2.00;
input double BasketRecoveryR                  = 0.25;
input double EmergencyIndividualStopATR       = 3.00;
input double TrailStartR                      = 2.00;
input double TrailATR                         = 1.00;
input int    MaxHoldMinutes                   = 45;
input int    ATRPeriod                        = 14;
input int    FastEMA                          = 12;
input int    SlowEMA                          = 144;
input int    RSIPeriod                        = 9;
input double MaxSpreadATR                     = 0.15;
input double DailyDrawdownLimitPct            = 5.0;
input double TotalDrawdownLimitPct            = 15.0;
input int    MaxConsecutiveLosses             = 2;
input int    LossHaltHours                    = 24;
input bool   EmergencyFlattenOnHardHalt        = true;
input int    FridayStopHourUTC                = 21;
// AE Math supervisor: each control can shrink/stop exposure only.
input double DDRCapPct                         = 15.0;
input double DDRGamma                          = 2.0;
input double DDVelocityLambda                  = 2.5;
input double DDAccelerationLambda              = 1.0;
input double MinFirstPassageRecoveryProbability= 0.55;
input int    FirstPassageLookback              = 30;
input int    SlippagePoints                    = 30;

CTrade trade;
string symbol_name;
int atr_handle = INVALID_HANDLE;
int ema_fast_handle = INVALID_HANDLE;
int ema_slow_handle = INVALID_HANDLE;
int rsi_handle = INVALID_HANDLE;
double peak_equity = 0.0;
double day_start_equity = 0.0;
datetime day_anchor = 0;
datetime halt_until = 0;
double previous_dd_pct = 0.0;
double previous_dd_velocity = 0.0;
datetime last_buy_add = 0;
datetime last_sell_add = 0;

//+------------------------------------------------------------------+
double Clamp(const double value,const double lo,const double hi)
{
   return MathMax(lo,MathMin(hi,value));
}

//+------------------------------------------------------------------+
bool ReadBufferValue(const int handle,const int shift,double &value)
{
   double buffer[];
   ArraySetAsSeries(buffer,true);
   if(CopyBuffer(handle,0,shift,1,buffer)!=1)
      return false;
   value=buffer[0];
   return true;
}

//+------------------------------------------------------------------+
double CurrentATR()
{
   double value=0.0;
   return ReadBufferValue(atr_handle,0,value) ? value : 0.0;
}

//+------------------------------------------------------------------+
double NormalizeVolume(double volume)
{
   double min_lot=SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MIN);
   double max_lot=SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MAX);
   double step=SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_STEP);
   if(step<=0.0)
      return 0.0;
   volume=Clamp(volume,min_lot,max_lot);
   volume=MathFloor(volume/step)*step;
   return NormalizeDouble(volume,2);
}

//+------------------------------------------------------------------+
bool IsFridayStop()
{
   MqlDateTime utc;
   TimeToStruct(TimeGMT(),utc);
   return (utc.day_of_week==5 && utc.hour>=FridayStopHourUTC);
}

//+------------------------------------------------------------------+
void RefreshDayAnchor()
{
   MqlDateTime now;
   TimeToStruct(TimeGMT(),now);
   now.hour=0; now.min=0; now.sec=0;
   datetime today=StructToTime(now);
   if(today!=day_anchor)
   {
      day_anchor=today;
      day_start_equity=AccountInfoDouble(ACCOUNT_EQUITY);
   }
}

//+------------------------------------------------------------------+
double TotalDrawdownPct()
{
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   if(peak_equity<=0.0)
      peak_equity=equity;
   peak_equity=MathMax(peak_equity,equity);
   return (peak_equity>0.0 ? 100.0*MathMax(0.0,peak_equity-equity)/peak_equity : 0.0);
}

//+------------------------------------------------------------------+
double DailyDrawdownPct()
{
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   return (day_start_equity>0.0 ? 100.0*MathMax(0.0,day_start_equity-equity)/day_start_equity : 0.0);
}

//+------------------------------------------------------------------+
int ConsecutiveLosses()
{
   if(!HistorySelect(TimeCurrent()-86400*30,TimeCurrent()))
      return 0;
   int losses=0;
   for(int i=HistoryDealsTotal()-1;i>=0;i--)
   {
      ulong ticket=HistoryDealGetTicket(i);
      if(ticket==0 || HistoryDealGetString(ticket,DEAL_SYMBOL)!=symbol_name)
         continue;
      long entry=HistoryDealGetInteger(ticket,DEAL_ENTRY);
      if(entry!=DEAL_ENTRY_OUT)
         continue;
      double pnl=HistoryDealGetDouble(ticket,DEAL_PROFIT)+HistoryDealGetDouble(ticket,DEAL_SWAP)+HistoryDealGetDouble(ticket,DEAL_COMMISSION);
      if(pnl>=0.0)
         break;
      losses++;
      if(losses>=MaxConsecutiveLosses)
         break;
   }
   return losses;
}

//+------------------------------------------------------------------+
bool HardHalt()
{
   if(DailyDrawdownPct()>=DailyDrawdownLimitPct || TotalDrawdownPct()>=TotalDrawdownLimitPct)
      return true;
   if(ConsecutiveLosses()>=MaxConsecutiveLosses)
   {
      halt_until=TimeCurrent()+LossHaltHours*3600;
      return true;
   }
   return (TimeCurrent()<halt_until);
}

//+------------------------------------------------------------------+
// Dynamic Drawdown Reserve x velocity/acceleration guard.
// Return is constrained to [0,1] and therefore cannot amplify a position.
double AEMathMultiplier()
{
   double dd=TotalDrawdownPct();
   if(dd>=DDRCapPct || DDRCapPct<=0.0)
      return 0.0;
   double reserve=MathPow(MathMax(0.0,1.0-dd/DDRCapPct),DDRGamma);
   double velocity=MathMax(0.0,dd-previous_dd_pct);
   double acceleration=MathMax(0.0,velocity-previous_dd_velocity);
   double guard=MathExp(-DDVelocityLambda*velocity-DDAccelerationLambda*acceleration);
   previous_dd_pct=dd;
   previous_dd_velocity=velocity;
   return Clamp(MathMin(reserve,guard),0.0,1.0);
}

//+------------------------------------------------------------------+
// Drift/volatility proxy for the chance that price returns to the basket mean.
// This is a gate, not a claim that the original EA used this model.
double FirstPassageProbability(const ENUM_POSITION_TYPE side,const double mean_price)
{
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int wanted=MathMax(FirstPassageLookback+1,5);
   if(CopyRates(symbol_name,PERIOD_M1,0,wanted,rates)<wanted)
      return 0.0;
   double sum=0.0, sum_sq=0.0;
   for(int i=0;i<FirstPassageLookback;i++)
   {
      double r=(rates[i].close-rates[i+1].close)/MathMax(rates[i+1].close,_Point);
      sum+=r; sum_sq+=r*r;
   }
   double mu=sum/FirstPassageLookback;
   double variance=MathMax(1e-12,sum_sq/FirstPassageLookback-mu*mu);
   double sigma=MathSqrt(variance);
   MqlTick tick;
   if(!SymbolInfoTick(symbol_name,tick))
      return 0.0;
   double price=(side==POSITION_TYPE_BUY ? tick.bid : tick.ask);
   double distance=(side==POSITION_TYPE_BUY ? mean_price-price : price-mean_price)/MathMax(price,_Point);
   if(distance<=0.0)
      return 1.0;
   double directional=(side==POSITION_TYPE_BUY ? mu : -mu);
   // Bounded logistic proxy: only used as a conservative add gate.
   double z=(directional*FirstPassageLookback-distance)/(sigma*MathSqrt(FirstPassageLookback));
   return Clamp(1.0/(1.0+MathExp(-z)),0.0,1.0);
}

//+------------------------------------------------------------------+
int BasketCount(const ulong magic,const ENUM_POSITION_TYPE side,double &volume,double &mean_price,datetime &first_open)
{
   int count=0;
   volume=0.0; mean_price=0.0; first_open=0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket==0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol_name || (ulong)PositionGetInteger(POSITION_MAGIC)!=magic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE)!=side)
         continue;
      double lots=PositionGetDouble(POSITION_VOLUME);
      volume+=lots;
      mean_price+=lots*PositionGetDouble(POSITION_PRICE_OPEN);
      datetime opened=(datetime)PositionGetInteger(POSITION_TIME);
      if(first_open==0 || opened<first_open)
         first_open=opened;
      count++;
   }
   if(volume>0.0)
      mean_price/=volume;
   return count;
}

//+------------------------------------------------------------------+
bool EntrySignal(const ENUM_POSITION_TYPE side)
{
   double fast=0.0, slow=0.0, rsi=0.0;
   if(!ReadBufferValue(ema_fast_handle,0,fast) || !ReadBufferValue(ema_slow_handle,0,slow) || !ReadBufferValue(rsi_handle,0,rsi))
      return false;
   // Deliberately a replaceable research proxy; it is NOT inferred original logic.
   return (side==POSITION_TYPE_BUY ? (fast>slow && rsi>=50.0) : (fast<slow && rsi<=50.0));
}

//+------------------------------------------------------------------+
bool OpenOrder(const ENUM_POSITION_TYPE side,const ulong magic,const double lots,const double atr)
{
   if(lots<=0.0 || atr<=0.0)
      return false;
   MqlTick tick;
   if(!SymbolInfoTick(symbol_name,tick))
      return false;
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(SlippagePoints);
   double price=(side==POSITION_TYPE_BUY ? tick.ask : tick.bid);
   // Profit is managed at basket level. The individual SL is only an outage/gap
   // backstop; normal loss control is the tighter basket hard-stop in ManageBasket.
   double sl=(side==POSITION_TYPE_BUY ? price-EmergencyIndividualStopATR*HardStopATR*atr : price+EmergencyIndividualStopATR*HardStopATR*atr);
   return (side==POSITION_TYPE_BUY ? trade.Buy(lots,symbol_name,0.0,sl,0.0,"TickScalper-G1-Buy")
                                    : trade.Sell(lots,symbol_name,0.0,sl,0.0,"TickScalper-G1-Sell"));
}

//+------------------------------------------------------------------+
void CloseBasket(const ulong magic,const ENUM_POSITION_TYPE side)
{
   trade.SetExpertMagicNumber(magic);
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket==0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL)==symbol_name && (ulong)PositionGetInteger(POSITION_MAGIC)==magic && (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE)==side)
         trade.PositionClose(ticket);
   }
}

//+------------------------------------------------------------------+
void ManageBasket(const ENUM_POSITION_TYPE side,const ulong magic)
{
   double lots,mean_price;
   datetime opened;
   int count=BasketCount(magic,side,lots,mean_price,opened);
   if(count==0)
      return;
   double atr=CurrentATR();
   if(atr<=0.0)
      return;
   MqlTick tick;
   if(!SymbolInfoTick(symbol_name,tick))
      return;
   double price=(side==POSITION_TYPE_BUY ? tick.bid : tick.ask);
   double r=(side==POSITION_TYPE_BUY ? price-mean_price : mean_price-price)/(HardStopATR*atr);
   if(r<=-1.0 || (TimeCurrent()-opened)>=MaxHoldMinutes*60)
   {
      CloseBasket(magic,side);
      return;
   }
   // A multi-leg basket exits together just above its weighted average price.
   // A single-leg trade retains the larger TargetR objective.
   double target_r=(count>1 ? BasketRecoveryR : TargetR);
   if(r>=target_r)
   {
      CloseBasket(magic,side);
      return;
   }
   // Maximum two adds, never geometric; the AE supervisor only shrinks BaseLots.
   int adds=count-1;
   double spread=(tick.ask-tick.bid)/atr;
   double fp=FirstPassageProbability(side,mean_price);
   datetime last_add=(side==POSITION_TYPE_BUY ? last_buy_add : last_sell_add);
   if(adds<MaxAdds && lots<BaseLots*MaxTotalRiskR && r<0.0 && EntrySignal(side) && spread<=MaxSpreadATR && fp>=MinFirstPassageRecoveryProbability && (TimeCurrent()-last_add)>=60)
   {
      // This cap includes the initial entry: total nominal risk can never exceed MaxTotalRiskR.
      double remaining=MathMax(0.0,BaseLots*MaxTotalRiskR-lots);
      double next=NormalizeVolume(MathMin(BaseLots*AEMathMultiplier(),remaining));
      if(next>0.0)
      {
         if(OpenOrder(side,magic,next,atr))
         {
            if(side==POSITION_TYPE_BUY)
               last_buy_add=TimeCurrent();
            else
               last_sell_add=TimeCurrent();
         }
      }
   }
   // ATR trail is only tightened after sufficient favorable movement.
   if(r>=TrailStartR)
   {
      double new_sl=(side==POSITION_TYPE_BUY ? price-TrailATR*atr : price+TrailATR*atr);
      for(int i=PositionsTotal()-1;i>=0;i--)
      {
         ulong ticket=PositionGetTicket(i);
         if(ticket==0 || !PositionSelectByTicket(ticket))
            continue;
         if(PositionGetString(POSITION_SYMBOL)!=symbol_name || (ulong)PositionGetInteger(POSITION_MAGIC)!=magic || (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE)!=side)
            continue;
         double old_sl=PositionGetDouble(POSITION_SL);
         bool improve=(side==POSITION_TYPE_BUY ? (old_sl==0.0 || new_sl>old_sl) : (old_sl==0.0 || new_sl<old_sl));
         if(improve)
            trade.PositionModify(ticket,new_sl,PositionGetDouble(POSITION_TP));
      }
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   symbol_name=(TradeSymbol=="" ? _Symbol : TradeSymbol);
   atr_handle=iATR(symbol_name,PERIOD_M1,ATRPeriod);
   ema_fast_handle=iMA(symbol_name,PERIOD_M1,FastEMA,0,MODE_EMA,PRICE_CLOSE);
   ema_slow_handle=iMA(symbol_name,PERIOD_M1,SlowEMA,0,MODE_EMA,PRICE_CLOSE);
   rsi_handle=iRSI(symbol_name,PERIOD_M1,RSIPeriod,PRICE_CLOSE);
   if(atr_handle==INVALID_HANDLE || ema_fast_handle==INVALID_HANDLE || ema_slow_handle==INVALID_HANDLE || rsi_handle==INVALID_HANDLE)
      return INIT_FAILED;
   peak_equity=AccountInfoDouble(ACCOUNT_EQUITY);
   RefreshDayAnchor();
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(atr_handle!=INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(ema_fast_handle!=INVALID_HANDLE) IndicatorRelease(ema_fast_handle);
   if(ema_slow_handle!=INVALID_HANDLE) IndicatorRelease(ema_slow_handle);
   if(rsi_handle!=INVALID_HANDLE) IndicatorRelease(rsi_handle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   RefreshDayAnchor();
   if(HardHalt())
   {
      if(EmergencyFlattenOnHardHalt)
      {
         CloseBasket(BuyMagic,POSITION_TYPE_BUY);
         CloseBasket(SellMagic,POSITION_TYPE_SELL);
      }
      return;
   }
   ManageBasket(POSITION_TYPE_BUY,BuyMagic);
   ManageBasket(POSITION_TYPE_SELL,SellMagic);
   if(!EnableTrading || IsFridayStop())
      return;
   double atr=CurrentATR();
   MqlTick tick;
   if(atr<=0.0 || !SymbolInfoTick(symbol_name,tick) || (tick.ask-tick.bid)/atr>MaxSpreadATR)
      return;
   double v,price; datetime opened;
   if(BasketCount(BuyMagic,POSITION_TYPE_BUY,v,price,opened)==0 && EntrySignal(POSITION_TYPE_BUY))
      OpenOrder(POSITION_TYPE_BUY,BuyMagic,NormalizeVolume(BaseLots),atr);
   if(BasketCount(SellMagic,POSITION_TYPE_SELL,v,price,opened)==0 && EntrySignal(POSITION_TYPE_SELL))
      OpenOrder(POSITION_TYPE_SELL,SellMagic,NormalizeVolume(BaseLots),atr);
}
//+------------------------------------------------------------------+
