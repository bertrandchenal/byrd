from baker import base_cli, run_batch


def test_all_conf(cfg, nominal_log, log_handler):
    cli = base_cli(['--dry-run'])
    cli.cfg = cfg
    for task in cfg.tasks.values():
        run_batch(task, [], cli)

    assert nominal_log == log_handler.getvalue()
