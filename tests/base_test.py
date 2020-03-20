import os
from shlex import shlex

import pytest

from byrd.main import run_batch
from byrd.cli import  load_cli
from byrd.utils import Env


def test_all_conf(test_cfg, log_buff):
    verbose = pytest.config.getoption('verbose', 0) > 0

    args = []
    if test_cfg.cli:
        lexer = shlex(test_cfg.cli)
        lexer.wordchars += '.!=<>:{}-/'
        args = list(lexer)
    cli = load_cli(args)

    base_env = Env(
        cli.env,
        cli.cfg.get('env'),
        os.environ,
    )

    for task in cli.tasks:
        run_batch(task, cli.hosts, cli, base_env)

    actual_lines = log_buff.getvalue().splitlines()
    actual_lines = [l.strip() for l in filter(None, actual_lines)]
    expected_output = test_cfg.output or ''
    expected_lines = expected_output.splitlines()
    expected_lines = [l.strip() for l in filter(None, expected_lines)]

    # Join and makes path independant of OS
    act = '\n'.join(actual_lines).replace('\\', '/')
    exp = '\n'.join(expected_lines).replace('\\', '/')
    if verbose:
        print(act)
    assert act == exp
