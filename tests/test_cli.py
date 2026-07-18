from __future__ import annotations

from chatcaht.cli import build_parser, main


def test_cli_selftest() -> None:
    assert main(["selftest"]) == 0


def test_cli_can_manage_audio_runtime_independently() -> None:
    args = build_parser().parse_args(["services", "status", "--only", "audio"])
    assert args.only == ["audio"]
