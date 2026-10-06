#property strict
#property version   "1.50"
#property description "AMOS Video Parity CISD Router v1.5 - M15 AMD + Normal CISD state machine"

#include <Trade/Trade.mqh>
CTrade trade;

enum ContextGateMode
  {
   CONTEXT_INFO_ONLY=0,
   CONTEXT_MACRO_ONLY=1,
   CONTEXT_PDA_VOL=2,
   CONTEXT_VIDEO_OR=3
  };

enum RouteType
  {
   ROUTE_NORMAL=0,
   ROUTE_AMD_ASIA=1,
   ROUTE_AMD_LONDON=2,
   ROUTE_AMD_NY=3
  };

enum PhaseType
  {
   PH_IDLE=0,
   PH_WAIT_SWEEP=1,
   PH_WAIT_CISD_CONFIRM=2,
   PH_WAIT_CISD_ENTRY=3
  };

input string          InpSymbol="XAUUSD";
input ulong           InpMagic=2026100715;
input double          InpRiskPct=0.35;
input int             InpATRPeriod=14;
input double          InpSweepMinATR=0.03;
input double          InpSLBufferATR=0.05;
input int             InpSweepTTL=8;
input int             InpConfirmTTL=8;
input int             InpEntryTTL=6;
input int             InpNormalLiquidityLookback=16;
input int             InpCISDLookback=8;
input double          InpMinTargetRR=0.80;
input double          InpVolumeInfluxMult=1.15;
input ContextGateMode InpContextMode=CONTEXT_INFO_ONLY;
input int             InpServerUTCOffsetHours=0;
input bool            InpOnePositionOnly=true;
input int             InpMaxTradesPerDay=8;
input double          InpMaxDailyLossPct=3.5;
input int             InpMaxSpreadPoints=0;
input int             InpDeviationPoints=30;
input bool            InpDrawChecklist=true;
input bool            InpWriteCSV=true;
input string          InpCSV="AMOS_VideoParity_CISD_Router_v1_5.csv";
input bool            InpPrintDiagnostics=true;

struct ChecklistState
  {
   bool liquiditySweep;
   bool htfPDA;
   bool cisdCandle;
   bool macroWindow;
   bool cisdConfirmed;
   bool volumeInflux;
   bool cisdEntry;
   bool clearTargets;
  };

string g_symbol;
datetime g_lastM15=0;
PhaseType g_phase=PH_IDLE;
RouteType g_route=ROUTE_NORMAL;
int g_age=0;
int g_dir=0;
double g_accHi=0.0,g_accLo=0.0,g_sweepExtreme=0.0,g_cisdLevel=0.0;
bool g_pda=false,g_macro=false,g_volume=false;
ChecklistState g_chk;
int g_csv=INVALID_HANDLE;
string g_prefix="AMOS_VP15_";

datetime ToJST(datetime server_time){return server_time-InpServerUTCOffsetHours*3600+9*3600;}
int MinuteOfDayJST(datetime server_time){MqlDateTime x;TimeToStruct(ToJST(server_time),x);return x.hour*60+x.min;}
int DateKeyJST(datetime server_time){MqlDateTime x;TimeToStruct(ToJST(server_time),x);return x.year*10000+x.mon*100+x.day;}
bool InRangeMinutes(int m,int a,int b){if(a<=b)return(m>=a&&m<b);return(m>=a||m<b);}
RouteType RouteAt(datetime t){int m=MinuteOfDayJST(t);if(InRangeMinutes(m,540,600))return ROUTE_AMD_ASIA;if(InRangeMinutes(m,960,1080))return ROUTE_AMD_LONDON;if(InRangeMinutes(m,1380,60))return ROUTE_AMD_NY;return ROUTE_NORMAL;}
bool IsMacroWindow(RouteType r,datetime t){int m=MinuteOfDayJST(t);if(r==ROUTE_AMD_ASIA)return InRangeMinutes(m,540,570);if(r==ROUTE_AMD_LONDON)return InRangeMinutes(m,960,990);if(r==ROUTE_AMD_NY)return InRangeMinutes(m,1380,1410);return false;}
string RouteName(RouteType r){if(r==ROUTE_AMD_ASIA)return"AMD_ASIA";if(r==ROUTE_AMD_LONDON)return"AMD_LONDON";if(r==ROUTE_AMD_NY)return"AMD_NY";return"NORMAL";}

bool LoadRates(ENUM_TIMEFRAMES tf,int n,MqlRates &r[]){ArraySetAsSeries(r,true);return CopyRates(g_symbol,tf,0,n,r)>=n;}
double ATR15(int shift=1){int h=iATR(g_symbol,PERIOD_M15,InpATRPeriod);if(h==INVALID_HANDLE)return 0.0;double a[];ArraySetAsSeries(a,true);double v=0.0;if(CopyBuffer(h,0,0,shift+2,a)>=shift+1)v=a[shift];IndicatorRelease(h);return v;}
bool NewM15Bar(){datetime t=iTime(g_symbol,PERIOD_M15,0);if(t<=0||t==g_lastM15)return false;g_lastM15=t;return true;}

bool SessionRangeForRoute(RouteType r,datetime now,double &hi,double &lo)
  {
   MqlRates b[];if(!LoadRates(PERIOD_M15,320,b))return false;int today=DateKeyJST(now);hi=-DBL_MAX;lo=DBL_MAX;bool found=false;
   if(r==ROUTE_NORMAL){int n=MathMin(InpNormalLiquidityLookback,ArraySize(b)-2);for(int i=2;i<2+n;i++){hi=MathMax(hi,b[i].high);lo=MathMin(lo,b[i].low);found=true;}return found;}
   for(int i=1;i<ArraySize(b);i++)
     {
      int key=DateKeyJST(b[i].time),m=MinuteOfDayJST(b[i].time);bool use=false;
      if(r==ROUTE_AMD_LONDON)use=(key==today&&m>=540&&m<960);
      else if(r==ROUTE_AMD_NY)use=(key==today&&m>=960&&m<1380);
      else if(r==ROUTE_AMD_ASIA){int prevKey=DateKeyJST(now-86400);use=((key==prevKey&&m>=1380)||(key==today&&m<360));}
      if(use){hi=MathMax(hi,b[i].high);lo=MathMin(lo,b[i].low);found=true;}
     }
   return found;
  }

bool CandleMatchesDelivery(const MqlRates &x,int direction){if(direction<0)return x.close>x.open;return x.close<x.open;}
bool BuildVideoCISDLevel(int direction,MqlRates &r[],double &level)
  {
   int j=1;if(!CandleMatchesDelivery(r[j],direction))j=2;if(j>=ArraySize(r)||!CandleMatchesDelivery(r[j],direction))return false;
   int oldest=j,maxShift=MathMin(ArraySize(r)-1,j+InpCISDLookback);for(int k=j+1;k<=maxShift;k++){if(!CandleMatchesDelivery(r[k],direction))break;oldest=k;}level=r[oldest].open;return true;
  }

bool HTFPDA(int direction,double px){MqlRates r[];if(!LoadRates(PERIOD_M30,26,r))return false;double hi=-DBL_MAX,lo=DBL_MAX;for(int i=2;i<26;i++){hi=MathMax(hi,r[i].high);lo=MathMin(lo,r[i].low);}double mid=(hi+lo)*0.5;return(direction<0?px>=mid:px<=mid);}
bool VolumeInflux(MqlRates &r[]){int n=MathMin(20,ArraySize(r)-2);if(n<8)return false;double v[];ArrayResize(v,n);for(int i=0;i<n;i++)v[i]=(double)r[i+2].tick_volume;ArraySort(v);double med=(n%2==1?v[n/2]:(v[n/2-1]+v[n/2])*0.5);return med>0&&(double)r[1].tick_volume>=med*InpVolumeInfluxMult;}
bool ContextAllows(){if(InpContextMode==CONTEXT_INFO_ONLY)return true;if(InpContextMode==CONTEXT_MACRO_ONLY)return g_macro;if(InpContextMode==CONTEXT_PDA_VOL)return(g_pda&&g_volume);return(g_macro||(g_pda&&g_volume));}

bool RRValid(int direction,double entry,double sl,double tp,double minRR,double &rr){double risk=MathAbs(entry-sl);if(risk<=0){rr=0;return false;}if(direction>0&&!(sl<entry&&tp>entry)){rr=0;return false;}if(direction<0&&!(sl>entry&&tp<entry)){rr=0;return false;}rr=MathAbs(tp-entry)/risk;return rr>=minRR;}
void ConsiderTarget(int direction,double entry,double sl,double cand,double &best,double &bestDist,double &bestRR,bool &found){double rr=0;if(!RRValid(direction,entry,sl,cand,InpMinTargetRR,rr))return;double d=MathAbs(cand-entry);if(d<bestDist){bestDist=d;best=cand;bestRR=rr;found=true;}}
bool FindClearTarget(int direction,double entry,double sl,double &tp,double &rr)
  {
   bool found=false;double best=0,bestDist=DBL_MAX,bestRR=0;MqlRates m15[];if(!LoadRates(PERIOD_M15,36,m15))return false;
   for(int i=3;i<MathMin(28,ArraySize(m15)-1);i++){if(direction<0&&m15[i].low<m15[i-1].low&&m15[i].low<m15[i+1].low&&m15[i].low<entry)ConsiderTarget(direction,entry,sl,m15[i].low,best,bestDist,bestRR,found);if(direction>0&&m15[i].high>m15[i-1].high&&m15[i].high>m15[i+1].high&&m15[i].high>entry)ConsiderTarget(direction,entry,sl,m15[i].high,best,bestDist,bestRR,found);}
   MqlRates m30[];if(LoadRates(PERIOD_M30,50,m30)){for(int i=2;i<MathMin(42,ArraySize(m30)-2);i++){if(m30[i].low>m30[i+2].high){double lo=m30[i+2].high,hi=m30[i].low;if(direction>0){ConsiderTarget(direction,entry,sl,lo,best,bestDist,bestRR,found);ConsiderTarget(direction,entry,sl,hi,best,bestDist,bestRR,found);}}if(m30[i].high<m30[i+2].low){double lo=m30[i].high,hi=m30[i+2].low;if(direction<0){ConsiderTarget(direction,entry,sl,hi,best,bestDist,bestRR,found);ConsiderTarget(direction,entry,sl,lo,best,bestDist,bestRR,found);}}}}
   ConsiderTarget(direction,entry,sl,(direction<0?g_accLo:g_accHi),best,bestDist,bestRR,found);if(found){tp=best;rr=bestRR;return true;}return false;
  }

bool HasOpenPosition(){for(int i=PositionsTotal()-1;i>=0;i--){ulong tk=PositionGetTicket(i);if(tk==0||!PositionSelectByTicket(tk))continue;if(PositionGetString(POSITION_SYMBOL)==g_symbol&&(ulong)PositionGetInteger(POSITION_MAGIC)==InpMagic)return true;}return false;}
datetime DayStart(datetime t){MqlDateTime x;TimeToStruct(t,x);x.hour=0;x.min=0;x.sec=0;return StructToTime(x);}
bool RiskGate(){if(InpOnePositionOnly&&HasOpenPosition())return false;if(InpMaxSpreadPoints>0&&(int)SymbolInfoInteger(g_symbol,SYMBOL_SPREAD)>InpMaxSpreadPoints)return false;int entries=0;double pnl=0;HistorySelect(DayStart(TimeCurrent()),TimeCurrent());for(int i=0;i<HistoryDealsTotal();i++){ulong d=HistoryDealGetTicket(i);if(!d)continue;if(HistoryDealGetString(d,DEAL_SYMBOL)!=g_symbol||(ulong)HistoryDealGetInteger(d,DEAL_MAGIC)!=InpMagic)continue;long e=HistoryDealGetInteger(d,DEAL_ENTRY);if(e==DEAL_ENTRY_IN)entries++;if(e==DEAL_ENTRY_OUT||e==DEAL_ENTRY_OUT_BY)pnl+=HistoryDealGetDouble(d,DEAL_PROFIT)+HistoryDealGetDouble(d,DEAL_SWAP)+HistoryDealGetDouble(d,DEAL_COMMISSION);}if(entries>=InpMaxTradesPerDay)return false;double bal=AccountInfoDouble(ACCOUNT_BALANCE);if(bal>0&&pnl<=-bal*InpMaxDailyLossPct/100.0)return false;return true;}
double NormalizeVolume(double x){double mn=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_MIN),mx=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_MAX),st=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_STEP);if(st<=0)st=mn;x=MathMax(mn,MathMin(mx,x));x=MathFloor(x/st+1e-9)*st;return NormalizeDouble(x,3);}
double RiskLots(int direction,double entry,double sl){double money=AccountInfoDouble(ACCOUNT_EQUITY)*InpRiskPct/100.0,p=0;ENUM_ORDER_TYPE typ=(direction>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL);if(money<=0||!OrderCalcProfit(typ,g_symbol,1.0,entry,sl,p)||MathAbs(p)<=0)return 0;return NormalizeVolume(money/MathAbs(p));}
bool PlaceEntry(int direction,double sl,double tp,string tag){if(!RiskGate())return false;MqlTick t;if(!SymbolInfoTick(g_symbol,t))return false;double entry=(direction>0?t.ask:t.bid),rr=0;if(!RRValid(direction,entry,sl,tp,0.10,rr))return false;double lot=RiskLots(direction,entry,sl);if(lot<=0)return false;trade.SetExpertMagicNumber(InpMagic);trade.SetDeviationInPoints(InpDeviationPoints);bool ok=(direction>0?trade.Buy(lot,g_symbol,0,sl,tp,tag):trade.Sell(lot,g_symbol,0,sl,tp,tag));if(InpPrintDiagnostics)Print("AMOS VP entry ",tag," rr=",DoubleToString(rr,2)," ok=",ok," ",trade.ResultRetcodeDescription());return ok;}

string Mark(bool v){return v?"[OK]":"[--]";} color MarkColor(bool v){return v?clrLimeGreen:clrTomato;}
void SetLabel(string id,string txt,int y,color c){string n=g_prefix+id;if(ObjectFind(0,n)<0)ObjectCreate(0,n,OBJ_LABEL,0,0,0);ObjectSetInteger(0,n,OBJPROP_CORNER,CORNER_RIGHT_LOWER);ObjectSetInteger(0,n,OBJPROP_ANCHOR,ANCHOR_RIGHT_LOWER);ObjectSetInteger(0,n,OBJPROP_XDISTANCE,12);ObjectSetInteger(0,n,OBJPROP_YDISTANCE,y);ObjectSetInteger(0,n,OBJPROP_COLOR,c);ObjectSetInteger(0,n,OBJPROP_FONTSIZE,9);ObjectSetString(0,n,OBJPROP_FONT,"Arial");ObjectSetString(0,n,OBJPROP_TEXT,txt);ObjectSetInteger(0,n,OBJPROP_SELECTABLE,false);}
void DrawChecklist(){if(!InpDrawChecklist)return;int y=185,s=18;SetLabel("00","Setup Checklist  "+RouteName(g_route),y,clrSilver);y-=s;SetLabel("01","Liquidity Sweep   "+Mark(g_chk.liquiditySweep),y,MarkColor(g_chk.liquiditySweep));y-=s;SetLabel("02","HTF PDA Delivery  "+Mark(g_chk.htfPDA),y,MarkColor(g_chk.htfPDA));y-=s;SetLabel("03","CISD Candle       "+Mark(g_chk.cisdCandle),y,MarkColor(g_chk.cisdCandle));y-=s;SetLabel("04","Macro Window      "+Mark(g_chk.macroWindow),y,MarkColor(g_chk.macroWindow));y-=s;SetLabel("05","CISD Confirmed    "+Mark(g_chk.cisdConfirmed),y,MarkColor(g_chk.cisdConfirmed));y-=s;SetLabel("06","Volume Influx     "+Mark(g_chk.volumeInflux),y,MarkColor(g_chk.volumeInflux));y-=s;SetLabel("07","CISD Entry        "+Mark(g_chk.cisdEntry),y,MarkColor(g_chk.cisdEntry));y-=s;SetLabel("08","Clear Targets     "+Mark(g_chk.clearTargets),y,MarkColor(g_chk.clearTargets));ChartRedraw();}
void LogState(string phase,bool fired=false,double tp=0,double rr=0){DrawChecklist();if(g_csv==INVALID_HANDLE)return;FileWrite(g_csv,TimeToString(iTime(g_symbol,PERIOD_M15,1),TIME_DATE|TIME_MINUTES),RouteName(g_route),phase,g_dir,(int)g_chk.liquiditySweep,(int)g_chk.htfPDA,(int)g_chk.cisdCandle,(int)g_chk.macroWindow,(int)g_chk.cisdConfirmed,(int)g_chk.volumeInflux,(int)g_chk.cisdEntry,(int)g_chk.clearTargets,(int)fired,DoubleToString(g_cisdLevel,_Digits),DoubleToString(tp,_Digits),DoubleToString(rr,3));FileFlush(g_csv);}
void ResetSetup(){g_phase=PH_IDLE;g_age=0;g_dir=0;g_accHi=0;g_accLo=0;g_sweepExtreme=0;g_cisdLevel=0;g_pda=false;g_macro=false;g_volume=false;ZeroMemory(g_chk);}

void ProcessVideoParity()
  {
   MqlRates r[];if(!LoadRates(PERIOD_M15,80,r))return;MqlRates b=r[1];double a=ATR15(1);if(a<=0)return;
   if(g_phase==PH_IDLE){g_route=RouteAt(b.time);if(!SessionRangeForRoute(g_route,b.time,g_accHi,g_accLo))return;g_phase=PH_WAIT_SWEEP;g_age=0;ZeroMemory(g_chk);g_chk.macroWindow=IsMacroWindow(g_route,b.time);LogState("ACCUMULATION");return;}
   g_age++;
   if(g_phase==PH_WAIT_SWEEP){bool sh=(b.high>g_accHi+a*InpSweepMinATR&&b.close<g_accHi),sl=(b.low<g_accLo-a*InpSweepMinATR&&b.close>g_accLo);if(sh||sl){g_dir=(sh?-1:+1);g_sweepExtreme=(sh?b.high:b.low);if(!BuildVideoCISDLevel(g_dir,r,g_cisdLevel)){ResetSetup();return;}g_pda=HTFPDA(g_dir,b.close);g_macro=IsMacroWindow(g_route,b.time);g_chk.liquiditySweep=true;g_chk.htfPDA=g_pda;g_chk.cisdCandle=true;g_chk.macroWindow=g_macro;g_phase=PH_WAIT_CISD_CONFIRM;g_age=0;LogState("SWEEP_CISD_CANDLE");return;}if(g_age>InpSweepTTL)ResetSetup();return;}
   if(g_phase==PH_WAIT_CISD_CONFIRM){if(g_dir<0)g_sweepExtreme=MathMax(g_sweepExtreme,b.high);else g_sweepExtreme=MathMin(g_sweepExtreme,b.low);bool confirmed=(g_dir<0?b.close<g_cisdLevel:b.close>g_cisdLevel);if(confirmed){g_volume=VolumeInflux(r);g_chk.cisdConfirmed=true;g_chk.volumeInflux=g_volume;g_phase=PH_WAIT_CISD_ENTRY;g_age=0;LogState("CISD_CONFIRMED");return;}if(g_age>InpConfirmTTL)ResetSetup();return;}
   if(g_phase==PH_WAIT_CISD_ENTRY){bool touched=(g_dir<0?b.high>=g_cisdLevel:b.low<=g_cisdLevel),held=(g_dir<0?b.close<=g_cisdLevel:b.close>=g_cisdLevel);if(touched&&held){g_chk.cisdEntry=true;MqlTick t;if(!SymbolInfoTick(g_symbol,t)){ResetSetup();return;}double entry=(g_dir>0?t.ask:t.bid),sl=(g_dir<0?g_sweepExtreme+a*InpSLBufferATR:g_sweepExtreme-a*InpSLBufferATR),tp=0,rr=0;g_chk.clearTargets=FindClearTarget(g_dir,entry,sl,tp,rr);LogState("CISD_ENTRY_READY",false,tp,rr);if(g_chk.clearTargets&&ContextAllows()){if(PlaceEntry(g_dir,sl,tp,RouteName(g_route)+"_CISD")){LogState("ENTRY",true,tp,rr);ResetSetup();return;}}}if(g_age>InpEntryTTL)ResetSetup();return;}
  }

int OnInit(){g_symbol=(InpSymbol==""?_Symbol:InpSymbol);trade.SetExpertMagicNumber(InpMagic);ResetSetup();if(InpWriteCSV){g_csv=FileOpen(InpCSV,FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI,',');if(g_csv!=INVALID_HANDLE){if(FileSize(g_csv)==0)FileWrite(g_csv,"time","route","phase","dir","sweep","htf_pda","cisd_candle","macro","cisd_confirmed","volume","cisd_entry","clear_targets","entry_fired","cisd_level","target","rr");FileSeek(g_csv,0,SEEK_END);}}return INIT_SUCCEEDED;}
void OnDeinit(const int reason){if(g_csv!=INVALID_HANDLE)FileClose(g_csv);for(int i=ObjectsTotal(0)-1;i>=0;i--){string n=ObjectName(0,i);if(StringFind(n,g_prefix)==0)ObjectDelete(0,n);}}
void OnTick(){if(_Symbol!=g_symbol&&!SymbolSelect(g_symbol,true))return;if(!NewM15Bar())return;ProcessVideoParity();}
