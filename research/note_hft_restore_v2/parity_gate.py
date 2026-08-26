from dataclasses import dataclass


@dataclass(frozen=True)
class Fingerprint:
    n: int = 176_483
    buy: int = 88_223
    sell: int = 88_260
    wr_pct: float = 72.71
    pf: float = 1.74
    maxdd_pct: float = 3.97


BASE = Fingerprint()


def evaluate(*, n: int, buy: int, sell: int, wr_pct=None, pf=None, maxdd_pct=None):
    n_ret = n / BASE.n if BASE.n else 0.0
    side_total = buy + sell
    buy_share = buy / side_total if side_total else 0.0
    base_buy_share = BASE.buy / (BASE.buy + BASE.sell)

    structural = {
        "n_retention": n_ret,
        "n_99_pass": n_ret >= 0.99,
        "buy_share": buy_share,
        "base_buy_share": base_buy_share,
        "side_share_abs_delta": abs(buy_share - base_buy_share),
        "side_balance_pass": side_total > 0 and abs(buy_share - base_buy_share) <= 0.01,
    }
    structural["pass"] = structural["n_99_pass"] and structural["side_balance_pass"]

    economic = {
        "wr_pct": wr_pct,
        "pf": pf,
        "maxdd_pct": maxdd_pct,
        "available": wr_pct is not None and pf is not None and maxdd_pct is not None,
    }
    if economic["available"]:
        economic["wr_delta_pp"] = wr_pct - BASE.wr_pct
        economic["pf_delta"] = pf - BASE.pf
        economic["dd_delta_pp"] = maxdd_pct - BASE.maxdd_pct

    return {
        "baseline": BASE.__dict__,
        "structural": structural,
        "economic": economic,
        "reality_noise_required_after_structural_pass": True,
    }
