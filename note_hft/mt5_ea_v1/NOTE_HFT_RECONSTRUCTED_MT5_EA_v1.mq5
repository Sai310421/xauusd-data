#property strict
#property version   "1.00"
#include <Trade/Trade.mqh>

CTrade trade;

input ulong   InpMagic               = 17648301;
input double  InpFixedLot            = 0.01;
input int     InpAlphaWindow         = 5;
input double  InpAlphaThreshold      = 0.0;
input int     InpCreateCooldownMs    = 3000;
input int     InpMinHoldMs           = 100;
input int     InpTimerMs             = 10;
input double  InpEntrySpreadMaxPts   = 0.0;
input double  InpMinFreeMarginRatio  = 0.40;
input bool    InpEnableTrading       = true;
input bool    InpWriteAuditCSV       = true;

// Optional reality sensitivity. Keep zero for Frozen baseline.
input double  InpRejectProbability   = 0.0;
input int     InpExtraEntryDelayMs   = 0;
input int     InpExtraCloseDelayMs   = 0;

double ask_buf[];
double bid_buf[];
int sample_count=0;
int ring_pos=0;
double a_cond_num=0.0;
double b_cond_num=0.0;

ulong last_entry_request_msc=0;
ulong next_create_msc=0;
bool entry_pending=false;
bool close_pending=false;
int last_entry_side=0;
long audit_handle=INVALID_HANDLE;

ulong NowMs(){ return (ulong)GetTickCount64(); }

bool OurPositionExists()
{
   if(!PositionSelect(_Symbol)) return false;
   return ((ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic);
}

double SpreadPoints()
{
   MqlTick t;
   if(!SymbolInfoTick(_Symbol,t)) return DBL_MAX;
   double pt=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   if(pt<=0.0) return DBL_MAX;
   return (t.ask-t.bid)/pt;
}

bool MarginGateOK()
{
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   double free=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(bal<=0.0) return false;
   return (free/bal)>InpMinFreeMarginRatio;
}

void OpenAudit()
{
   if(!InpWriteAuditCSV) return;
   string fn="NOTE_HFT_AUDIT_"+_Symbol+".csv";
   audit_handle=FileOpen(fn,FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON|FILE_SHARE_WRITE,',');
   if(audit_handle==INVALID_HANDLE) return;
   if(FileSize(audit_handle)==0)
      FileWrite(audit_handle,"time_msc","event","side","bid","ask","spread_pts","a_cond","b_cond","position_price","profit","hold_ms");
   else
      FileSeek(audit_handle,0,SEEK_END);
}

void Audit(string ev,int side=0,double pos_price=0.0,double profit=0.0,long hold_ms=0)
{
   if(!InpWriteAuditCSV || audit_handle==INVALID_HANDLE) return;
   MqlTick t; if(!SymbolInfoTick(_Symbol,t)) return;
   double pt=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   double sp=(pt>0.0 ? (t.ask-t.bid)/pt : 0.0);
   FileWrite(audit_handle,(long)t.time_msc,ev,side,DoubleToString(t.bid,_Digits),DoubleToString(t.ask,_Digits),DoubleToString(sp,3),DoubleToString(a_cond_num,_Digits+2),DoubleToString(b_cond_num,_Digits+2),DoubleToString(pos_price,_Digits),DoubleToString(profit,2),hold_ms);
   FileFlush(audit_handle);
}

bool SyntheticReject()
{
   if(InpRejectProbability<=0.0) return false;
   if(InpRejectProbability>=1.0) return true;
   return ((double)MathRand()/32767.0)<InpRejectProbability;
}

void PushQuote(double ask,double bid)
{
   if(ArraySize(ask_buf)!=InpAlphaWindow)
   {
      ArrayResize(ask_buf,InpAlphaWindow);
      ArrayResize(bid_buf,InpAlphaWindow);
      ArrayInitialize(ask_buf,0.0);
      ArrayInitialize(bid_buf,0.0);
      sample_count=0; ring_pos=0;
   }

   ask_buf[ring_pos]=ask;
   bid_buf[ring_pos]=bid;
   ring_pos=(ring_pos+1)%InpAlphaWindow;
   if(sample_count<InpAlphaWindow) sample_count++;
   if(sample_count<InpAlphaWindow){ a_cond_num=0.0; b_cond_num=0.0; return; }

   int oldest=ring_pos;
   int newest=(ring_pos-1+InpAlphaWindow)%InpAlphaWindow;
   double old_ask=ask_buf[oldest];
   double old_bid=bid_buf[oldest];
   double new_ask=ask_buf[newest];
   double new_bid=bid_buf[newest];

   // Reconstructed missing fragment. Frozen entry conditions are unchanged.
   a_cond_num=old_ask-new_ask;
   b_cond_num=new_bid-old_bid;
}

int CurrentSignal()
{
   if(sample_count<InpAlphaWindow) return 0;
   bool buy=(b_cond_num>InpAlphaThreshold && a_cond_num<0.0);
   bool sell=(a_cond_num>InpAlphaThreshold && b_cond_num<0.0);
   if(buy && !sell) return 1;
   if(sell && !buy) return -1;
   return 0;
}

bool EntryGateOK()
{
   if(!InpEnableTrading || OurPositionExists() || entry_pending || close_pending) return false;
   if(NowMs()<next_create_msc) return false;
   if(!MarginGateOK()) return false;
   if(SpreadPoints()>InpEntrySpreadMaxPts+1e-12) return false;
   return true;
}

void TryEntry()
{
   if(!EntryGateOK()) return;
   int sig=CurrentSignal(); if(sig==0) return;
   if(SyntheticReject()){ Audit("SYNTH_REJECT",sig); return; }
   if(InpExtraEntryDelayMs>0) Sleep(InpExtraEntryDelayMs);

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetAsyncMode(true);
   last_entry_request_msc=NowMs();
   next_create_msc=last_entry_request_msc+(ulong)MathMax(0,InpCreateCooldownMs);
   last_entry_side=sig;
   entry_pending=true;

   bool ok=(sig>0 ? trade.Buy(InpFixedLot,_Symbol,0,0,0,"NOTE_HFT_BUY") : trade.Sell(InpFixedLot,_Symbol,0,0,0,"NOTE_HFT_SELL"));
   if(!ok){ entry_pending=false; Audit("ENTRY_SEND_FAIL",sig); }
   else Audit("ENTRY_SENT",sig);
}

void TryClose()
{
   if(!OurPositionExists() || close_pending) return;
   ulong min_close=last_entry_request_msc+(ulong)MathMax(0,InpMinHoldMs)+(ulong)MathMax(0,InpExtraCloseDelayMs);
   if(NowMs()<min_close) return;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetAsyncMode(true);
   double p=PositionGetDouble(POSITION_PRICE_OPEN);
   double pnl=PositionGetDouble(POSITION_PROFIT);
   long hold=(long)(NowMs()-last_entry_request_msc);
   close_pending=true;
   if(!trade.PositionClose(_Symbol)){ close_pending=false; Audit("CLOSE_SEND_FAIL",last_entry_side,p,pnl,hold); }
   else Audit("CLOSE_SENT",last_entry_side,p,pnl,hold);
}

int OnInit()
{
   if(InpAlphaWindow<2 || InpFixedLot<=0.0) return INIT_PARAMETERS_INCORRECT;
   MathSrand((uint)TimeLocal());
   ArrayResize(ask_buf,InpAlphaWindow); ArrayResize(bid_buf,InpAlphaWindow);
   ArrayInitialize(ask_buf,0.0); ArrayInitialize(bid_buf,0.0);
   trade.SetExpertMagicNumber(InpMagic); trade.SetAsyncMode(true);
   OpenAudit();
   EventSetMillisecondTimer((int)MathMax(1,InpTimerMs));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(audit_handle!=INVALID_HANDLE){ FileFlush(audit_handle); FileClose(audit_handle); audit_handle=INVALID_HANDLE; }
}

void OnTick()
{
   MqlTick t; if(!SymbolInfoTick(_Symbol,t)) return;
   PushQuote(t.ask,t.bid);
   // Frozen close-first invariant.
   if(OurPositionExists()){ TryClose(); return; }
   TryEntry();
}

void OnTimer()
{
   if(OurPositionExists()) TryClose();
}

void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
{
   if(trans.symbol!=_Symbol || trans.type!=TRADE_TRANSACTION_DEAL_ADD || trans.deal==0) return;
   if((ulong)HistoryDealGetInteger(trans.deal,DEAL_MAGIC)!=InpMagic) return;

   long et=HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
   double price=HistoryDealGetDouble(trans.deal,DEAL_PRICE);
   double profit=HistoryDealGetDouble(trans.deal,DEAL_PROFIT);
   if(et==DEAL_ENTRY_IN)
   {
      entry_pending=false; close_pending=false;
      Audit("ENTRY_FILLED",last_entry_side,price,0.0,0);
   }
   else if(et==DEAL_ENTRY_OUT || et==DEAL_ENTRY_OUT_BY)
   {
      long hold=(long)(NowMs()-last_entry_request_msc);
      Audit("EXIT_FILLED",last_entry_side,price,profit,hold);
      entry_pending=false; close_pending=false; last_entry_side=0;
   }
}
