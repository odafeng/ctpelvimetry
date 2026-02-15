"""Tests for ctpelvimetry.cli."""

import pytest

from ctpelvimetry.cli import main


class TestMainEntryPoint:

    def test_no_args_shows_help(self):
        """Running with no args should exit (no subcommand)."""
        with pytest.raises(SystemExit):
            main()

    def test_help_flag(self):
        """--help should exit with code 0."""
        import sys
        old_argv = sys.argv
        sys.argv = ["ctpelvimetry", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv


class TestPelvSubcommand:

    def test_pelv_help(self):
        """pelv --help should exit with code 0."""
        import sys
        old_argv = sys.argv
        sys.argv = ["ctpelvimetry", "pelv", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv


class TestBodyCompSubcommand:

    def test_body_comp_help(self):
        """body-comp --help should exit with code 0."""
        import sys
        old_argv = sys.argv
        sys.argv = ["ctpelvimetry", "body-comp", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
