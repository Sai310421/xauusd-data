#property copyright "AMOS research candidate"
#property version "0.10"
#property strict
#include <Trade/Trade.mqh>
CTrade trade;

// TickScalper reconstruction candidate.
// TickSmoother v2.1 is SOURCE-DERIVED, not exact original parity.
// Current Nautilus KPI used synthetic accounting; MT5 uses broker-native P/L/equity.
// Cashback is ledger-only; it never changes trading decisions.

input ulong MagicNumber=260701;
input double BaseLot=0.30;
input int MaxLayers=10;
input double AddDistance=3.11;
input double BasketOffset=0.8575;
input double EmergencyDistance=3.80;
input int SessionStartHourUTC=7, SessionEndHourUTC=17, CooldownSeconds=45;
input int TicksPerBar=5, FastMA=3, SlowMA=5, ConfirmMA1=8, ConfirmMA2=13;
input bool CrossOnly=true;
input double DDRCapPct=28.0, DDRLambda=3.0, DDRMinScale=0.50;
input int DDRStartLayer=8;
input double FastRescueFraction=0.06, RescueHardFDDPct=45.0;
input double FastBEDistance=6.20, FastFDDThreshold=19.0, FastVolThreshold=0.12;
input double CashbackUSDPerLot=6.0;

struct VLeg{double price;double lot;};
VLeg legs[];
double closes[],deltas[],fdds[];
int side=0,tick_count=0,prev_signal=0;
bool rescue=false,have_mid=false;
double prev_mid=0,peak_eq=0,cb_lots=0,cb_usd=0,ddr_min=1.0;
datetime last_close=0;
long ddr_scaled=0,rescue_started=0,rescue_completed=0,rescue_denied=0;

void Push(double &a[],double v,int maxn){int n=ArraySize(a);if(n<maxn){ArrayResize(a,n+1);a[n]=v;return;}for(int i=1;i<n;i++)a[i-1]=a[i];a[n-1]=v;}
double FloorLot(double x){double st=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP),mn=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN),mx=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);if(st<=0)st=.01;double v=MathFloor((x+1e-12)/st)*st;return MathMax(mn,MathMin(mx,NormalizeDouble(v,8)));}
double RawLot(int n){if(n==0)return FloorLot(BaseLot);if(n==1)return FloorLot(BaseLot*1.4666666667);return FloorLot(BaseLot*MathPow(1.5,n));}
double Gross(){double q=0;for(int i=0;i<ArraySize(legs);i++)q+=legs[i].lot;return q;}
double BE(){double q=Gross(),s=0;if(q<=0)return 0;for(int i=0;i<ArraySize(legs);i++)s+=legs[i].price*legs[i].lot;return s/q;}
double FDD(){double e=AccountInfoDouble(ACCOUNT_EQUITY);peak_eq=MathMax(peak_eq,e);return peak_eq>0?100.0*(peak_eq-e)/peak_eq:0;}
double Mean(int n){int z=ArraySize(closes);if(z<n)return EMPTY_VALUE;double s=0;for(int i=z-n;i<z;i++)s+=closes[i];return s/n;}

int Signal(double mid){
 tick_count++;if(tick_count<TicksPerBar)return 0;tick_count=0;Push(closes,mid,256);
 if(ArraySize(closes)<ConfirmMA2)return 0;
 double f=Mean(FastMA),s=Mean(SlowMA),c1=Mean(ConfirmMA1),c2=Mean(ConfirmMA2);int raw=0;
 if(mid>f&&f>s&&s>c1&&c1>c2)raw=1;else if(mid<f&&f<s&&s<c1&&c1<c2)raw=-1;
 if(CrossOnly){int fire=(raw!=0&&raw!=prev_signal)?raw:0;prev_signal=raw;return fire;}prev_signal=raw;return raw;
}
void Update(double bid,double ask){
 double e=AccountInfoDouble(ACCOUNT_EQUITY);peak_eq=MathMax(peak_eq,e);Push(fdds,FDD(),1000);
 double m=(bid+ask)/2;if(have_mid)Push(deltas,m-prev_mid,400);prev_mid=m;have_mid=true;
}
double DDR(){
 double d=FDD(),dist=MathMax(0.0,DDRCapPct-d),x=MathMin(1.0,dist/MathMax(DDRCapPct,1e-12));
 double den=1.0-MathExp(-DDRLambda),raw=den>1e-12?(1.0-MathExp(-DDRLambda*x))/den:x;
 double sc=MathMax(DDRMinScale,MathMin(1.0,raw));if(sc<.999)ddr_scaled++;ddr_min=MathMin(ddr_min,sc);return sc;
}
double Vol(){
 int n=ArraySize(deltas);if(n<80)return 999;int st=MathMax(0,n-200),k=n-st;double mu=0;
 for(int i=st;i<n;i++)mu+=deltas[i]*side;mu/=k;double v=0;for(int i=st;i<n;i++){double z=deltas[i]*side-mu;v+=z*z;}return MathSqrt(MathMax(v/MathMax(k-1,1),0.0));
}
bool FastOK(double bid,double ask){double mark=side>0?bid:ask,bed=(BE()-mark)*side;return bed<=FastBEDistance||(FDD()<FastFDDThreshold&&Vol()<=FastVolThreshold);}
bool Send(int s,double lot){lot=FloorLot(lot);if(lot<=0)return false;bool ok=s>0?trade.Buy(lot,_Symbol):trade.Sell(lot,_Symbol);if(!ok)Print("order fail ",trade.ResultRetcodeDescription());return ok;}
void Leg(double p,double l){int n=ArraySize(legs);ArrayResize(legs,n+1);legs[n].price=p;legs[n].lot=l;}
void Reset(){side=0;rescue=false;ArrayResize(legs,0);last_close=TimeCurrent();}
bool CloseAll(string why){if(PositionSelect(_Symbol)&&!trade.PositionClose(_Symbol)){Print("close fail ",why);return false;}Print("close ",why," FDD=",DoubleToString(FDD(),2)," CB=",DoubleToString(cb_usd,2));Reset();return true;}
bool Open(int s,double bid,double ask){double l=RawLot(0);if(!Send(s,l))return false;side=s;ArrayResize(legs,0);Leg(s>0?ask:bid,l);return true;}
bool Rescue(double bid,double ask){
 if(!FastOK(bid,ask)){rescue_denied++;return false;}double l=FloorLot(Gross()*FastRescueFraction);
 if(!Send(side,l))return false;Leg(side>0?ask:bid,l);rescue=true;rescue_started++;return true;
}
void Manage(double bid,double ask){
 if(rescue){if(PositionSelect(_Symbol)&&PositionGetDouble(POSITION_PROFIT)>=0){rescue_completed++;CloseAll("FAST_COMPLETE");return;}if(FDD()>=RescueHardFDDPct)CloseAll("FAST_HARD_FDD");return;}
 double mark=side>0?bid:ask,be=BE();if((mark-be)*side>=BasketOffset){CloseAll("BASKET");return;}
 int n=ArraySize(legs);double last=legs[n-1].price,adv=side>0?last-mark:mark-last;
 if(n<MaxLayers&&adv>=AddDistance){double raw=RawLot(n),sc=n>=DDRStartLayer?DDR():1.0,l=FloorLot(raw*sc);if(l>0&&Send(side,l))Leg(side>0?ask:bid,l);return;}
 if(n>=MaxLayers&&adv>=EmergencyDistance){if(!Rescue(bid,ask))CloseAll("EMERGENCY_FAST_DENIED");}
}
int OnInit(){trade.SetExpertMagicNumber(MagicNumber);trade.SetTypeFillingBySymbol(_Symbol);peak_eq=AccountInfoDouble(ACCOUNT_EQUITY);return INIT_SUCCEEDED;}
void OnTick(){
 MqlTick t;if(!SymbolInfoTick(_Symbol,t))return;Update(t.bid,t.ask);
 if(ArraySize(legs)==0){
  if(PositionSelect(_Symbol))return;if(last_close>0&&TimeCurrent()-last_close<CooldownSeconds)return;
  MqlDateTime tm;TimeToStruct(TimeGMT(),tm);if(tm.hour<SessionStartHourUTC||tm.hour>=SessionEndHourUTC)return;
  int s=Signal((t.bid+t.ask)/2);if(s!=0)Open(s,t.bid,t.ask);return;
 }
 Manage(t.bid,t.ask);
}
void OnTradeTransaction(const MqlTradeTransaction &tr,const MqlTradeRequest &rq,const MqlTradeResult &rs){
 if(tr.type!=TRADE_TRANSACTION_DEAL_ADD||tr.deal==0||!HistoryDealSelect(tr.deal))return;
 if((ulong)HistoryDealGetInteger(tr.deal,DEAL_MAGIC)!=MagicNumber||HistoryDealGetString(tr.deal,DEAL_SYMBOL)!=_Symbol)return;
 ENUM_DEAL_ENTRY e=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(tr.deal,DEAL_ENTRY);
 if(e==DEAL_ENTRY_OUT||e==DEAL_ENTRY_OUT_BY){cb_lots+=HistoryDealGetDouble(tr.deal,DEAL_VOLUME);cb_usd=cb_lots*CashbackUSDPerLot;}
}
void OnDeinit(const int reason){Print("DDR_scaled=",ddr_scaled," DDR_min=",DoubleToString(ddr_min,3)," FAST=",rescue_started,"/",rescue_completed," denied=",rescue_denied," closed_lots=",DoubleToString(cb_lots,2)," cashback_est_USD=",DoubleToString(cb_usd,2));}
