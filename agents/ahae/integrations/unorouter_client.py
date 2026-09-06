"""UnoRouter adapter for AMOS/ODS.

Design goals:
- OpenAI-compatible chat/completions transport.
- No third-party Python dependency.
- Fail closed when API key is missing.
- Keep trading/backtest execution deterministic: this client is for analysis,
  review, hypothesis generation, and supervisor shadow workflows only.
"""

from __future__ import annotations

import json
import os
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
    timeout_seconds: int = 90

    @classmethod
    def from_env(cls) -> "UnoRouterConfig":
        api_key = os.getenv("UNOROUTER_API_KEY", "").strip()
        if not api_key:
            raise UnoRouterError("UNOROUTER_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=os.getenv("UNOROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout_seconds=int(os.getenv("UNOROUTER_TIMEOUT_SECONDS", "90")),
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
                "User-Agent": "AMOS-ODS-UnoRouter/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise UnoRouterError(f"UnoRouter HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise UnoRouterError(f"UnoRouter connection failed: {exc.reason}") from exc

    def text(self, **kwargs: Any) -> str:
        data = self.chat(**kwargs)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise UnoRouterError(f"Unexpected UnoRouter response shape: {data}") from exc


def smoke_test() -> None:
    model = os.getenv("UNOROUTER_SMOKE_MODEL", "glm-5.3-flash:free")
    client = UnoRouterClient()
    text = client.text(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return only the token AMOS_UNOROUTER_OK.",
            },
            {"role": "user", "content": "health check"},
        ],
        temperature=0.0,
        max_tokens=32,
    )
    if "AMOS_UNOROUTER_OK" not in text:
        raise UnoRouterError(f"Smoke test returned unexpected content: {text!r}")
    print(f"UnoRouter smoke test OK ({model})")


if __name__ == "__main__":
    smoke_test()
