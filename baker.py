from getpass import getpass
from hashlib import md5
from itertools import chain
from collections import ChainMap, OrderedDict, defaultdict
import argparse
import logging
import os
import posixpath
import sys

from fabric import Connection, Config
from invoke import run
from tanker import yaml_load
import invoke
import yaml


try:
    import keyring
except ImportError:
    keyring = None

__version__ = '0.0'

fmt = '%(levelname)s:%(asctime).19s: %(message)s'
logging.basicConfig(format=fmt)
logger = logging.getLogger('baker')
logger.setLevel(logging.INFO)

try:
    import colorama
    colorama.init()
    MAGENTA = colorama.Fore.MAGENTA
    RED = colorama.Fore.RED
    RESET = colorama.Style.RESET_ALL

    # We define custom handler ..
    class Handler(logging.StreamHandler):
        def format(self, record):
            if record.levelname == 'INFO':
                record.msg = MAGENTA + record.msg + RESET
            elif record.levelname in ('WARNING', 'ERROR', 'CRITICAL'):
                record.msg = RED + record.msg + RESET
            return super(Handler, self).format(record)

    #  .. and plug it
    handler = Handler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = 0
except ImportError:
    pass


def yaml_load(stream):
    class OrderedLoader(yaml.Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def edits(word):
    yield word
    splits = ((word[:i], word[i:]) for i in range(len(word) + 1))
    for left, right in splits:
        if right:
            yield left + right[1:]

def gen_candidates(wordlist):
    candidates = defaultdict(set)
    for word in wordlist:
        for ed1 in edits(word):
            for ed2 in edits(ed1):
                candidates[ed2].add(word)
    return candidates

def spell(candidates,  word):
    matches = set(chain.from_iterable(
        candidates[ed] for ed in edits(word) if ed in candidates
    ))
    return matches

def spellcheck(objdict, word):
    if word in objdict:
        return

    candidates = objdict.get('_candidates')
    if not candidates:
        candidates = gen_candidates(list(objdict))
        objdict['_candidates'] = candidates

    msg = '"%s" not found in %s' % (word, objdict._path)
    matches = spell(candidates, word)
    if matches:
        msg += ', try: %s' % ' or '.join(matches)
    logger.error(msg)
    exit(1)

class ObjectDict(dict):
    """
    Simple objet sub-class that allows to transform a dict into an
    object, like: `ObjectDict({'ham': 'spam'}).ham == 'spam'`
    """
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __iter__(self):
        for key in self.keys():
            if key.startswith('_'):
                continue
            yield key

class Node:

    @staticmethod
    def fail(path, kind):
        msg = 'Error while parsing config: expecting "%s" while parsing "%s"'
        logger.error(msg % (kind, '->'.join(path)))
        sys.exit()

    @classmethod
    def parse(cls, cfg, path=tuple()):
        children = getattr(cls, '_children', None)
        type_name = children and type(children).__name__ \
                    or ' or '.join((c.__name__ for c in cls._type))
        res = None
        if type_name == 'dict':
            if not isinstance(cfg, dict):
                cls.fail(path, type_name)
            res = ObjectDict()

            if '*' in children:
                assert len(children) == 1, "Don't mix '*' and other keys"
                child_class = children['*']
                for name, value in cfg.items():
                    res[name] = child_class.parse(value, path + (name,))
            else:
                # Enforce known pre-defined
                for key in cfg:
                    if key not in children:
                        path = ' -> '.join(path)
                        msg = 'Attribute "%s" not understoodin %s' % (key, path)
                        candidates = gen_candidates(children.keys())
                        matches = spell(candidates, key)
                        if matches:
                            msg += ', try: %s' % ' or '.join(matches)
                        logger.error(msg)
                        exit(1)

                for name, child_class in children.items():
                    if name not in cfg:
                        continue
                    res[name] = child_class.parse(cfg.pop(name), path + (name,))

        elif type_name == 'list':
            if not isinstance(cfg, list):
                cls.fail(path, type_name)
            child_class = children[0]
            res = [child_class.parse(c, path+ ('[]',)) for c in cfg]

        else:
            if not isinstance(cfg, cls._type):
                cls.fail(path, type_name)
            res = cfg

        return cls.setup(res, path)

    @classmethod
    def setup(cls, values, path):
        if isinstance(values, dict):
            values['_path'] = '->'.join(path)
        return values


class Atom(Node):
    _type = (str, bool)

class AtomList(Node):
    _children = [Atom]

class Hosts(Node):
    _children = [Atom]

class Auth(Node):
    _children = {'*': Atom}

class EnvNode(Node):
    _children = {'*': Atom}

class HostGroup(Node):
    _children = {
        'hosts': Hosts,
    }

class Network(Node):
    _children = {
        '*': HostGroup,
    }

class Multi(Node):
    _children = {
        'task': Atom,
        'export': Atom,
        'env': EnvNode,
    }

class MultiList(Node):
    _children = [Multi]

class Command(Node):
    _children = {
        'desc': Atom,
        'local': Atom,
        'once': Atom,
        'run': Atom,
        'env': EnvNode,
        'multi': MultiList,
    }

    @classmethod
    def setup(cls, values, path):
        if path:
            values['name'] = path[-1]
        if 'desc' not in values:
            values['desc'] = values['name']
        return values

class Task(Node):
    _children = {
        '*': Command,
    }

class LoadNode(Node):
    _children = {
        'file': Atom,
        'as': Atom,
    }

class LoadList(Node):
    _children = [LoadNode]

class ConfigRoot(Node):
    _children = {
        'networks': Network,
        'tasks': Task,
        'auth': Auth,
        'env': EnvNode,
        'load': LoadList,
    }


class Env(ChainMap):

    def __init__(self, *dicts):
        return super().__init__(*filter(bool, dicts))

    def fmt(self, string):
        try:
            return string.format(**self)
        except KeyError as exc:
            msg = 'Unable to format "%s" (missing: "%s")'% (string, exc.args[0])
            candidates = gen_candidates(self.keys())
            key = exc.args[0]
            matches = spell(candidates, key)
            if msg:
                msg += ', try: %s' % ' or '.join(matches)
            logger.error(msg )
            exit(1)
        except IndexError as exc:
            msg = 'Unable to format "%s", positional argument not supported'
            logger.error(msg)
            sys.exit()

def get_passphrase(key_path):
    service = 'SSH private key'
    csum = md5(open(key_path, 'rb').read()).digest().hex()
    ssh_pass = keyring.get_password(service, csum)
    if not ssh_pass:
        ssh_pass = getpass('Password for %s: ' % key_path)
        keyring.set_password(service, csum, ssh_pass)
    return ssh_pass


def get_sudo_passwd():
    service = "Sudo password"
    passwd = keyring.get_password(service, '-')
    if not passwd:
        passwd = getpass('Sudo password:')
        keyring.set_password(service, '-', passwd)
    return passwd


CONNECTION_CACHE = {}
def connect(host, auth, with_sudo=False):
    if (host, with_sudo) in CONNECTION_CACHE:
        return CONNECTION_CACHE[host, with_sudo]

    connect_kwargs = {}
    if auth and auth.get('ssh_private_key'):
        connect_kwargs['key_filename'] = auth.ssh_private_key
        if not os.path.exists(auth.ssh_private_key):
            msg = 'Private key file "%s" not found' % auth.ssh_private_key
            logger.error(msg)
            sys.exit()
        ssh_pass = get_passphrase(auth.ssh_private_key)
        connect_kwargs['password'] = ssh_pass

    if with_sudo:
        config = Config(overrides={
            'sudo': {
                'password': get_sudo_passwd()
            }
        })
    else:
        config = None

    con = Connection(host, config=config, connect_kwargs=connect_kwargs)
    CONNECTION_CACHE[host, with_sudo] = con
    return con


def run_local(task, env, cli):
    # Run local task
    cmd = env.fmt(task.local)
    # TODO log only task_desc and let desc contains env info like {host}
    logger.info(env.fmt('RUN {task_name} locally'))
    if cli.dry_run:
        logger.info('[DRY-RUN] ' + cmd)
        return None
    res = run(cmd, env=env)
    return res


def run_remote(task, host, env, cli):
    res = None
    host = env.fmt(host)
    con = connect(host, cli.cfg.auth, bool(task.sudo))
    if task.run:
        cmd = env.fmt(task.run)
        logger.info(env.fmt('RUN {task_name} ON {host}'))
        if cli.dry_run:
            logger.info('[DRY-RUN] ' + cmd)
        else:
            res = con.run(cmd, pty=True, env=env)

    elif task.sudo:
        cmd = env.fmt(task.sudo)
        logger.info(env.fmt('SUDO {task_name} ON {host}'))

        if cli.dry_run:
            logger.info('[DRY-RUN] %s' + cmd)
        else:
            res = con.sudo(cmd)

    elif task.send:
        local_path = env.fmt(task.send)
        remote_path = env.fmt(task.to)
        logger.info(f'SEND {local_path} TO {host}:{remote_path}')
        if cli.dry_run:
            logger.info('[DRY-RUN]')
        elif os.path.isfile(local_path):
            con.put(local_path, remote=remote_path)
        else:
            for root, subdirs, files in os.walk(local_path):
                rel_dir = os.path.relpath(root, local_path)
                rem_dir = posixpath.join(remote_path, rel_dir)
                con.run('mkdir -p {}'.format(rem_dir))
                for f in files:
                    rel_f = os.path.join(root, f)
                    rem_file = posixpath.join(rem_dir, f)
                    con.put(os.path.abspath(rel_f), remote=rem_file)
    else:
        logger.error('Unable to run task "%s"' % task.name)
        sys.exit()

    return res


def run_task(task, host, cli, parent_env=None):
    '''
    Execute one task on one host (or locally)
    '''

    # Prepare environment
    env = Env(
        # Cli is top priority
        dict(e.split('=') for e in cli.env),
        # Then comes env from parent task
        parent_env,
        # Env on the task itself
        task.get('env'),
        # Top-level env
        cli.cfg.get('env'),
        # OS env
        os.environ,
    ).new_child()
    env.update({
        'task_desc': env.fmt(task.desc),
        'task_name': task.name,
        'host': host or '',
    })

    if task.local:
        res = run_local(task, env, cli)
    else:
        res = run_remote(task, host, env, cli)

    if task.get('assert'):
        env.update({
            'stdout': res.stdout,
            'stderr': res.stderr,
        })
        assert_ = env.fmt(task['assert'])
        ok = eval(assert_, dict(env))
        if ok:
            logger.info('Assert ok')
        else:
            logger.error('Assert "%s" failed!' % assert_)
            sys.exit()
    return res


def run_batch(task, hosts, cli, env=None):
    '''
    Run one task on a list of hosts
    '''
    env = Env(task.get('env'), env)
    res = None
    export_env = {}

    if task.get('multi'):
        for multi in task.multi:
            task = multi.task
            spellcheck(cli.cfg.tasks, task)
            sub_task = cli.cfg.tasks[task]
            network = multi.get('network')
            if network:
                spellcheck(cli.cfg.networks, network)
                hosts = cli.cfg.networks[network].hosts
            child_env = multi.get('env', {}).copy()
            for k, v in child_env.items():
                # env wrap-around!
                child_env[k] = env.fmt(child_env[k])
            run_env = Env(export_env, child_env, env)
            res = run_batch(sub_task, hosts, cli, run_env)
            if multi.export:
                export_env[multi.export] = res and res.stdout.strip() or ''

    else:
        if task.once and task.local:
            res = run_task(task, None, cli, env)
            return res
        for host in hosts:
            res = run_task(task, host, cli, env)
            if task.once:
                break
    return res


def base_cli(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('names',  nargs='*',
                        help='Hosts and commands to run them on')
    parser.add_argument('-c', '--config', default='bk.yaml',
                        help='Config file')
    parser.add_argument('-r', '--run', nargs='*', default=[],
                        help='Run custom task')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='Do not run actual tasks, just print them')
    parser.add_argument('-e', '--env', nargs='*', default=[],
                        help='Add value to execution environment '
                        '(ex: -e foo=bar "name=John Doe")')
    parser.add_argument('-s', '--sudo', default='auto',
                        help='Enable sudo (auto|yes|no')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Increase verbosity')
    cli = parser.parse_args(args=args)
    return ObjectDict(vars(cli))


def load(path, prefix=None):
    load_sections = ('networks', 'tasks', 'auth', 'env')

    logger.info('Load config %s' % path)
    cfg = yaml_load(open(path))
    cfg = ConfigRoot.parse(cfg)

    # Define useful defaults
    cfg.networks = cfg.networks or ObjectDict()
    cfg.tasks = cfg.tasks or ObjectDict()

    if prefix:
        fn = lambda x: '/'.join(prefix + [x])
        # Apply prefix
        for section in load_sections:
            if not cfg.get(section):
                continue
            items = cfg[section].items()
            cfg[section] = {fn(k): v for k, v in items if not k.startswith('_')}

    # Recursive load
    if cfg.load:
        cfg_path = os.path.dirname(path)
        for item in cfg.load:
            if item.get('as'):
                child_prefix = item['as']
            else:
                child_prefix, _ = os.path.splitext(item.file)
            child_path = os.path.join(cfg_path, item.file)
            child_cfg = load(child_path, child_prefix.split('/'))

            for section in load_sections:
                if not cfg.get(section):
                    cfg[section] = {}
                cfg[section].update(child_cfg.get(section, {}))
    return cfg

def main():
    cli = base_cli()
    if cli.verbose:
        level = 'INFO' if cli.verbose == 1 else 'DEBUG'
        logger.setLevel(level)
        logger.info('Log level set to: %s' % level)

    # Load config
    cfg = load(cli.config)
    cli.cfg = cfg

    # Make sure we don't have overlap between hosts and tasks
    items = list(cfg.networks) + list(cfg.tasks)
    msg = 'Name collision between tasks and networks'
    assert len(set(items)) == len(items), msg


    # Build task list
    tasks = []
    networks = []
    for name in cli.names:
        if name in cfg.networks:
            host = cfg.networks[name]
            networks.append(host)
        elif name in cfg.tasks:
            task = cfg.tasks[name]
            tasks.append(task)
        else:
            msg = 'Name "%s" not understood' % name
            matches = spell(cfg.networks, name) | spell(cfg.tasks, name)
            if matches:
                msg += ', try: %s' % ' or '.join(matches)
            logger.error(msg)
            sys.exit()

    for custom_task in cli.run:
        task = Command.parse(yaml_load(custom_task))
        task.desc = 'Custom command'
        tasks.append(task)

    hosts = list(chain.from_iterable(n.hosts for n in networks))

    try:
        for task in tasks:
            run_batch(task, hosts, cli)
    except invoke.exceptions.Failure as e:
        logger.error(str(e))


if __name__ == '__main__':
    main()
