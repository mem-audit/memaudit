from __future__ import annotations

import pytest

from memaudit.cli import build_parser, main


def test_parser_requires_audit_args():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["audit"])
    ns = parser.parse_args(
        ["audit", "--model", "./out", "--canary-set", "./memaudit-canaries.json", "--dataset", "./train.jsonl"]
    )
    assert ns.ref == "auto"
    assert ns.model == "./out"


def test_main_help():
    # argparse --help exits; the wrapper may also return 0
    try:
        code = main(["--help"])
        assert code == 0
    except SystemExit as ei:
        assert ei.code == 0


def test_demo_help():
    try:
        code = main(["demo", "--help"])
        assert code == 0
    except SystemExit as ei:
        assert ei.code == 0


def test_audit_help():
    try:
        code = main(["audit", "--help"])
        assert code == 0
    except SystemExit as ei:
        assert ei.code == 0


def test_manifest_alias():
    parser = build_parser()
    ns = parser.parse_args(
        ["audit", "--model", "./out", "--manifest", "./memaudit-manifest.json"]
    )
    assert ns.canary_set == "./memaudit-manifest.json"
    assert ns.ref == "auto"


def test_missing_canary_set_is_human(tmp_path, capsys):
    code = main(["audit", "--model", str(tmp_path), "--canary-set", str(tmp_path / "nope.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert "memaudit:" in err
    assert "canary-set" in err or "manifest" in err
    assert "Traceback" not in err


def test_main_no_command_is_usage():
    assert main([]) == 2
