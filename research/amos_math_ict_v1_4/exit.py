from .models import ExitState, ExitDecision

class ExitRouter:
    def __init__(self,enable_structure_exit=True,enable_liquidity_take=True,enable_breakeven=True,enable_atr_trail=True,enable_time_stop=True,enable_ai_exit=False,be_arm_mfe_r=1.0,atr_trail_mult=1.5,time_stop_seconds=300.0,min_progress_ratio=0.25):
        self.enable_structure_exit=enable_structure_exit; self.enable_liquidity_take=enable_liquidity_take; self.enable_breakeven=enable_breakeven; self.enable_atr_trail=enable_atr_trail; self.enable_time_stop=enable_time_stop; self.enable_ai_exit=enable_ai_exit
        self.be_arm_mfe_r=be_arm_mfe_r; self.atr_trail_mult=atr_trail_mult; self.time_stop_seconds=time_stop_seconds; self.min_progress_ratio=min_progress_ratio
        self.be_armed=False

    def reset(self):
        self.be_armed=False

    def _close(self, reason: str) -> ExitDecision:
        self.reset()
        return ExitDecision('CLOSE', reason)

    def decide(self,s:ExitState)->ExitDecision:
        if self.enable_structure_exit and (s.structure_invalidated or s.opposite_mss or s.opposite_choch):
            return self._close('STRUCTURE_INVALIDATION')
        if self.enable_liquidity_take and s.liquidity_target_hit and s.pnl>0:
            return self._close('LIQUIDITY_TARGET')
        if self.enable_ai_exit and s.ai_reject_now and s.pnl>0:
            return self._close('AI_PROFIT_EXIT')
        if self.enable_breakeven and self.be_armed and s.pnl<=0:
            return self._close('ECONOMIC_BE')

        risk_unit=max(abs(s.mae),1e-12)
        just_armed=False
        if self.enable_breakeven and (not self.be_armed) and s.pnl>0 and s.mfe/risk_unit>=self.be_arm_mfe_r:
            self.be_armed=True
            just_armed=True

        if self.enable_atr_trail and s.pnl>0 and s.atr>0:
            trail_distance=self.atr_trail_mult*s.atr
            if s.mfe-s.pnl>=trail_distance:
                return self._close('ATR_TRAIL')

        if self.enable_time_stop and s.seconds_held>=self.time_stop_seconds:
            if s.mfe/(abs(s.mae)+1e-12)<self.min_progress_ratio:
                return self._close('TIME_NO_PROGRESS')

        if just_armed:
            return ExitDecision('PROTECT_BE','MFE_BE_ARM',lock_fraction=1.0)
        return ExitDecision('HOLD','NO_EXIT')
