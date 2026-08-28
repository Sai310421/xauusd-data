from __future__ import annotations

import inspect
import json
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest import BacktestEngine


def describe(module_name: str, names: list[str]) -> dict:
    out = {}
    try:
        mod = __import__(module_name, fromlist=names)
    except Exception as exc:
        return {"__import_error__": repr(exc)}
    for name in names:
        obj = getattr(mod, name, None)
        if obj is None:
            out[name] = {"available": False}
            continue
        try:
            sig = str(inspect.signature(obj))
        except Exception as exc:
            sig = f"UNAVAILABLE:{exc!r}"
        out[name] = {
            "available": True,
            "module": getattr(obj, "__module__", None),
            "signature": sig,
            "doc": (inspect.getdoc(obj) or "")[:1200],
        }
    return out


def main() -> None:
    report = {
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "execution": describe(
            "nautilus_trader.execution",
            [
                "FillModel",
                "ProbabilisticFillModel",
                "DefaultFillModel",
                "MakerTakerFeeModel",
                "FixedFeeModel",
                "PerContractFeeModel",
                "StaticLatencyModel",
            ],
        ),
        "backtest_models": describe(
            "nautilus_trader.backtest.models",
            ["FillModel", "MakerTakerFeeModel", "LatencyModel"],
        ),
    }
    try:
        report["BacktestEngine.add_venue_signature"] = str(inspect.signature(BacktestEngine.add_venue))
    except Exception as exc:
        report["BacktestEngine.add_venue_signature"] = f"UNAVAILABLE:{exc!r}"

    out = Path("results/native-reality-probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "api_probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
