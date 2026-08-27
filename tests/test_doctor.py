from memaudit.doctor import validate_report


def test_validate_report_catches_invalid_headline():
    bad = {
        "schema_version": "1.0.0",
        "tool_version": "0.1.0",
        "membership": {"tpr_at_1pct_fpr": 1.0, "headline_valid": True, "n_controls": 10},
        "regurgitation": {"overall": {"rate": 0}},
        "negative_controls": {"n": 10},
        "limitations": "x",
        "report_hash": "abc",
        "phone_home": False,
        "local_only": True,
    }
    fails = validate_report(bad)
    assert any("headline_valid" in f for f in fails)
