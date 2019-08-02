import argparse
import os

from config import yaml_load,ConfigRoot
from main import get_hosts_and_tasks, run_batch, info
from utils import (enable_logging_color, log_handler, logger, ByrdException,
                    ObjectDict, Env, abort)

basedir, _ = os.path.split(__file__)
PKG_DIR = os.path.join(basedir, 'pkg')



def load_cfg(path, prefix=None):
    load_sections = ('networks', 'tasks', 'auth', 'env')

    if os.path.isfile(path):
        logger.debug('Load config %s' % os.path.relpath(path))
        cfg = yaml_load(open(path))
        cfg = ConfigRoot.parse(cfg)
    else:
        raise ByrdException('Config file "%s" not found' % path)

    # Define useful defaults
    cfg.networks = cfg.networks or ObjectDict()
    cfg.tasks = cfg.tasks or ObjectDict()

    # Create backrefs between tasks to the local config
    if cfg.get('tasks'):
        cfg_cp = cfg.copy()
        for k, v in cfg['tasks'].items():
            v._cfg = cfg_cp


    # Recursive load
    if cfg.load:
        cfg_path = os.path.dirname(path)
        for item in cfg.load:
            if item.get('file'):
                rel_path = item.file
                child_path = os.path.join(cfg_path, item.file)
            elif item.get('pkg'):
                rel_path = item.pkg
                child_path = os.path.join(PKG_DIR, item.pkg)

            if item.get('as'):
                child_prefix = item['as']
            else:
                child_prefix, _ = os.path.splitext(rel_path)

            child_cfg = load_cfg(child_path, child_prefix)
            key_fn = lambda x: '/'.join([child_prefix, x])
            for section in load_sections:
                if not section in child_cfg:
                    continue
                items = {key_fn(k): v for k, v in child_cfg[section].items()}
                cfg[section].update(items)
    return cfg


def load_cli(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('names',  nargs='*',
                        help='Hosts and commands to run them on')
    parser.add_argument('-c', '--config', default='bd.yaml',
                        help='Config file')
    parser.add_argument('-R', '--run', nargs='*', default=[],
                        help='Run remote task')
    parser.add_argument('-L', '--run-local', nargs='*', default=[],
                        help='Run local task')
    parser.add_argument('-P', '--run-python', nargs='*', default=[],
                        help='Run python task')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='Do not run actual tasks, just print them')
    parser.add_argument('-e', '--env', nargs='*', default=[],
                        help='Add value to execution environment '
                        '(ex: -e foo=bar "name=John Doe")')
    parser.add_argument('-s', '--sudo', default='auto',
                        help='Enable sudo (auto|yes|no')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Increase verbosity')
    parser.add_argument('-q', '--quiet', action='count',
                        default=0, help='Decrease verbosity')
    parser.add_argument('-n', '--no-color', action='store_true',
                        help='Disable colored logs')
    parser.add_argument('-i', '--info', action='store_true',
                        help='Print info')
    cli = parser.parse_args(args=args)
    cli = ObjectDict(vars(cli))

    # Load config
    cfg = load_cfg(cli.config)
    cli.cfg = cfg
    cli.update(get_hosts_and_tasks(cli, cfg))

    # Transformt env string into dict
    cli.env = dict(e.split('=') for e in cli.env)
    return cli

def run():
    cli = None
    try:
        cli = load_cli()
        if not cli.no_color:
            enable_logging_color()
        cli.verbose = max(0, 1 + cli.verbose - cli.quiet)
        level = ['WARNING', 'INFO', 'DEBUG'][min(cli.verbose, 2)]
        logger.setLevel(level)
        log_handler.setLevel(level)

        if cli.info:
            info(cli)
            return

        base_env = Env(
            cli.env, # Highest-priority
            cli.cfg.get('env'),
            os.environ, # Lowest
        )
        for task in cli.tasks:
            run_batch(task, cli.hosts, cli, base_env)
    except ByrdException as e:
        if cli and cli.verbose > 2:
            raise
        abort(str(e))

if __name__ == '__main__':
    run()
