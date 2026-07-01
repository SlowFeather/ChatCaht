from __future__ import annotations

from chatcaht.cli import main


def test_cli_selftest() -> None:
    assert main(["selftest"]) == 0
