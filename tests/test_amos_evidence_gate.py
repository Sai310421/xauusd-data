from research.amos_evidence_gate import EVIDENCE_FIELDS, WR5_FIELDS, evaluate


def complete_payload():
    p = {k: "measured" for k in EVIDENCE_FIELDS if k != "wr5"}
    p["wr5"] = {k: "measured" for k in WR5_FIELDS}
    return p


def test_complete_evidence_is_eligible():
    r = evaluate(complete_payload())
    assert r["evidence_status"] == "VALID"
    assert r["wr5_status"] == "VALID_FOR_DECISION"
    assert r["promotion_eligible"] is True


def test_missing_wr5_is_invalid_not_reject():
    p = complete_payload()
    del p["wr5"]["slippage"]
    r = evaluate(p)
    assert r["wr5_status"] == "INVALID"
    assert r["promotion_eligible"] is False
    assert "slippage" in r["missing_wr5"]


def test_missing_core_evidence_blocks_promotion():
    p = complete_payload()
    del p["git_sha"]
    r = evaluate(p)
    assert r["evidence_status"] == "INVALID"
    assert r["promotion_eligible"] is False
