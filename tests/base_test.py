from baker import base_cli, run_batch


def test_all_conf(cfg):
    cli = base_cli(['--dry-run'])
    cli.cfg = cfg
    import pdb;pdb.set_trace()
    for task in cfg.tasks.values():
        run_batch(task, [], cli)
