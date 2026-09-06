"""UnoRouter adapter for AMOS/ODS.

Design goals:
- OpenAI-compatible chat/completions transport.
- No third-party Python dependency.
- Fail closed when API key is missing.
- Retry/fail over transient provider timeouts for smoke/advisory workloads.
- Keep trading/backtest execution deterministic: this client is for analysis,
  review, hypothesis generation, and supervisor shadow workflows only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://api.unorouter.com/v1"


class UnoRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class UnoRouterConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 45
    retries: int = 1

    @classmethod
    def from_env(cls) -> "UnoRouterConfig":
        api_key = os.getenv("UNOROUTER_API_KEY", "").strip()
        if not api_key:
            raise UnoRouterError("UNOROUTER_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=os.getenv("UNOROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout_seconds=int(os.getenv("UNOROUTER_TIMEOUT_SECONDS", "45")),
            retries=int(os.getenv("UNOROUTER_RETRIES", "1")),
        )


class UnoRouterClient:
    def __init__(self, config: Optional[UnoRouterConfig] = None) -> None:
        self.config = config or UnoRouterConfig.from_env()

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AMOS-ODS-UnoRouter/1.1",
            },
            method="POST",
        )

        last_error: Optional[BaseException] = None
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                # Authentication/configuration errors should fail immediately.
                if exc.code in (400, 401, 403, 404):
                    raise UnoRouterError(f"UnoRouter HTTP {exc.code}: {body[:1000]}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc

            if attempt < self.config.retries:
                time.sleep(2 ** attempt)

        raise UnoRouterError(
            f"UnoRouter request failed after {self.config.retries + 1} attempt(s): {last_error}"
        ) from last_error

    def text(self, **kwargs: Any) -> str:
        data = self.chat(**kwargs)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise UnoRouterError(f"Unexpected UnoRouter response shape: {data}") from exc


def smoke_test() -> None:
    requested = os.getenv("UNOROUTER_SMOKE_MODEL", "").strip()
    configured = os.getenv(
        "UNOROUTER_SMOKE_MODELS",
        "glm-5.3-flash:free,codestral-latest:free,glm-5.3-flash-thinking:free",
    )
    models: List[str] = []
    for model in ([requested] if requested else []) + configured.split(","):
        model = model.strip()
        if model and model not in models:
            models.append(model)

    client = UnoRouterClient()
    errors: List[str] = []
    for model in models:
        try:
            text = client.text(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only AMOS_UNOROUTER_OK."},
                    {"role": "user", "content": "health check"},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            if "AMOS_UNOROUTER_OK" in text:
                print(f"UnoRouter smoke test OK ({model})")
                return
            errors.append(f"{model}: unexpected response {text!r}")
        except UnoRouterError as exc:
            errors.append(f"{model}: {exc}")
            print(f"UnoRouter smoke candidate failed ({model}): {exc}")

    raise UnoRouterError("All smoke models failed: " + " | ".join(errors))


if __name__ == "__main__":
    smoke_test()
