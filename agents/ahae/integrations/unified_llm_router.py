"""Unified Smart LLM Router for AMOS/ODS advisory workloads.

Routes analysis/review/hypothesis tasks across Experiential Labs and UnoRouter.
Never call this module from deterministic execution paths that place trades or
mutate measured backtest results.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiential_client import ExperientialClient, ExperientialError
from unorouter_client import UnoRouterClient, UnoRouterError

POLICY_PATH = Path(__file__).with_name("unified_llm_routes.json")


class UnifiedRouterError(RuntimeError):
    pass


@dataclass
class GatewayState:
    failures: int = 0
    open_until: float = 0.0


class UnifiedLLMRouter:
    def __init__(self, policy_path: Path = POLICY_PATH) -> None:
        self.policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.states: Dict[str, GatewayState] = {
            "experiential": GatewayState(),
            "unorouter": GatewayState(),
        }
        self.failure_threshold = int(os.getenv("AMOS_LLM_CB_FAILURES", "2"))
        self.cooldown_seconds = int(os.getenv("AMOS_LLM_CB_COOLDOWN_SECONDS", "60"))

    def _model_for(self, gateway: str, role: str) -> str:
        cfg = self.policy["roles"][role][gateway]
        env_name = cfg.get("model_env")
        if env_name:
            configured = os.getenv(env_name, "").strip()
            if configured:
                return configured
        return cfg["default_model"]

    def _available(self, gateway: str) -> bool:
        return time.time() >= self.states[gateway].open_until

    def _success(self, gateway: str) -> None:
        state = self.states[gateway]
        state.failures = 0
        state.open_until = 0.0

    def _failure(self, gateway: str) -> None:
        state = self.states[gateway]
        state.failures += 1
        if state.failures >= self.failure_threshold:
            state.open_until = time.time() + self.cooldown_seconds

    def _call_gateway(
        self,
        gateway: str,
        *,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Tuple[str, str]:
        model = self._model_for(gateway, role)
        if gateway == "experiential":
            client = ExperientialClient()
            return client.text(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ), model
        if gateway == "unorouter":
            client = UnoRouterClient()
            return client.text(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ), model
        raise UnifiedRouterError(f"Unsupported gateway: {gateway}")

    def text(
        self,
        *,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if role not in self.policy["roles"]:
            raise UnifiedRouterError(f"Unknown role: {role}")

        errors: List[str] = []
        order = self.policy["roles"][role]["gateway_order"]
        for gateway in order:
            if gateway not in self.states:
                continue
            if not self._available(gateway):
                errors.append(f"{gateway}: circuit_open")
                continue
            try:
                text, model = self._call_gateway(
                    gateway,
                    role=role,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not isinstance(text, str) or not text.strip():
                    raise UnifiedRouterError("empty response")
                self._success(gateway)
                return {
                    "text": text,
                    "gateway": gateway,
                    "model": model,
                    "role": role,
                    "fallback_used": gateway != order[0],
                }
            except (ExperientialError, UnoRouterError, UnifiedRouterError) as exc:
                self._failure(gateway)
                errors.append(f"{gateway}: {exc}")

        raise UnifiedRouterError("All configured gateways failed: " + " | ".join(errors))


def smoke_test() -> None:
    router = UnifiedLLMRouter()
    result = router.text(
        role="cheap_worker",
        messages=[
            {"role": "system", "content": "Return only AMOS_UNIFIED_ROUTER_OK."},
            {"role": "user", "content": "health check"},
        ],
        temperature=0.0,
        max_tokens=32,
    )
    if "AMOS_UNIFIED_ROUTER_OK" not in result["text"]:
        raise UnifiedRouterError(f"Unexpected smoke response: {result}")
    print(
        "Unified router smoke OK "
        f"gateway={result['gateway']} model={result['model']} "
        f"fallback={result['fallback_used']}"
    )


if __name__ == "__main__":
    smoke_test()
