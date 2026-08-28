from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import nautilus_trader


def describe(module_name: str, names: list[str]) -> dict:
    out = {}
    try:
        mod = importlib.import_module(module_name)
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


def describe_attrs(module_name: str, class_name: str, tokens: tuple[str, ...]) -> dict:
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except Exception as exc:
        return {"__error__": repr(exc)}
    out = {}
    for name in sorted(n for n in dir(cls) if any(t in n.lower() for t in tokens)):
        obj = getattr(cls, name, None)
        try:
            sig = str(inspect.signature(obj))
        except Exception as exc:
            sig = f"UNAVAILABLE:{exc!r}"
        out[name] = {
            "signature": sig,
            "doc": (inspect.getdoc(obj) or "")[:800],
        }
    return out


def locate_backtest_engine() -> dict:
    candidates = [
        ("nautilus_trader.backtest", "BacktestEngine"),
        ("nautilus_trader.backtest.engine", "BacktestEngine"),
        ("nautilus_trader.backtest.node", "BacktestEngine"),
    ]
    attempts = []
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            obj = getattr(mod, attr, None)
            if obj is None:
                attempts.append({"module": module_name, "available": False})
                continue
            try:
                cls_sig = str(inspect.signature(obj))
            except Exception as exc:
                cls_sig = f"UNAVAILABLE:{exc!r}"
            try:
                add_venue_sig = str(inspect.signature(obj.add_venue))
            except Exception as exc:
                add_venue_sig = f"UNAVAILABLE:{exc!r}"
            return {
                "found": True,
                "module": module_name,
                "class_signature": cls_sig,
                "add_venue_signature": add_venue_sig,
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append({"module": module_name, "import_error": repr(exc)})
    return {"found": False, "attempts": attempts}


def main() -> None:
    report = {
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "backtest_engine": locate_backtest_engine(),
        "fusion_imports": {
            "common": describe("nautilus_trader.common", ["LogLevel"]),
            "common_enums": describe("nautilus_trader.common.enums", ["LogLevel"]),
            "config": describe("nautilus_trader.config", ["BacktestEngineConfig", "LoggerConfig", "LoggingConfig", "RiskEngineConfig"]),
            "model": describe("nautilus_trader.model", ["BarType", "Money", "Venue", "TraderId", "AccountType", "OmsType", "OrderSide"]),
            "model_data": describe("nautilus_trader.model.data", ["Bar", "QuoteTick", "BarType"]),
            "model_enums": describe("nautilus_trader.model.enums", ["AccountType", "OmsType", "OrderSide"]),
            "model_identifiers": describe("nautilus_trader.model.identifiers", ["InstrumentId", "TraderId", "Venue"]),
            "model_objects": describe("nautilus_trader.model.objects", ["Money"]),
            "trading_config": describe("nautilus_trader.trading.config", ["StrategyConfig"]),
            "trading_strategy": describe("nautilus_trader.trading.strategy", ["Strategy"]),
            "persistence_catalog": describe("nautilus_trader.persistence.catalog", ["ParquetDataCatalog"]),
        },
        "catalog_read_api": describe_attrs(
            "nautilus_trader.persistence.catalog",
            "ParquetDataCatalog",
            ("query", "quote", "instrument", "read"),
        ),
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
            ["FillModel", "MakerTakerFeeModel", "LatencyModel", "StaticLatencyModel"],
        ),
        "backtest_engine_module": describe(
            "nautilus_trader.backtest.engine",
            ["BacktestEngine", "BacktestVenueConfig"],
        ),
    }

    out = Path("results/native-reality-probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "api_probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["backtest_engine"].get("found"):
        raise SystemExit("BACKTEST_ENGINE_NOT_FOUND")


if __name__ == "__main__":
    main()
