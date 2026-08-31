//+------------------------------------------------------------------+
//| AMOS_EDGE_APEX_RECOGNITION_v1.0.mqh                              |
//| Reusable clean-room edge extracted from GORIRIN analysis          |
//+------------------------------------------------------------------+
#ifndef __AMOS_EDGE_APEX_RECOGNITION_V1_0__
#define __AMOS_EDGE_APEX_RECOGNITION_V1_0__

struct ApexEdgeInput
{
   int  rci_extreme_count;     // 0..3
   bool cci_extreme;
   bool rsi_extreme;
   bool cci_reversal;
   bool price_turn;
   bool deviation_confirmed;   // c3, only when exact formula is supplied
   bool deviation_enabled;
};

struct ApexEdgeResult
{
   double score;
   bool   apex_candidate;
   bool   high_confidence;
};

class CAMOSApexRecognitionEdge
{
public:
   ApexEdgeResult Evaluate(const ApexEdgeInput &x,
                           const double min_score=4.0,
                           const bool require_price_turn=true)
   {
      ApexEdgeResult r;
      r.score=0.0;

      if(x.rci_extreme_count>=2) r.score+=1.0;
      if(x.rci_extreme_count>=3) r.score+=1.0;
      if(x.cci_extreme)           r.score+=1.0;
      if(x.rsi_extreme)           r.score+=1.0;
      if(x.cci_reversal)          r.score+=1.0;
      if(x.price_turn)            r.score+=1.0;
      if(x.deviation_enabled && x.deviation_confirmed)
         r.score+=1.0;

      r.apex_candidate=(r.score>=min_score &&
                        (!require_price_turn || x.price_turn));

      r.high_confidence=(r.score>=MathMax(min_score+1.0,5.0) &&
                         x.rci_extreme_count>=2 &&
                         x.cci_reversal &&
                         x.price_turn);
      return r;
   }

   bool AllowNanpin(const ApexEdgeResult &r,
                    const int current_stage,
                    const int max_risk_stage,
                    const double min_nanpin_score=3.0)
   {
      if(current_stage>=max_risk_stage) return false;
      if(!r.apex_candidate) return false;
      return (r.score>=min_nanpin_score);
   }

   double ExpandedGrid(const double base_grid_pips,
                       const int current_positions,
                       const double expansion_per_stage=0.06)
   {
      int n=MathMax(0,current_positions-1);
      return base_grid_pips*MathPow(1.0+MathMax(0.0,expansion_per_stage),(double)n);
   }
};
#endif
