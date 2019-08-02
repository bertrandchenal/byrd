from getpass import getpass
from hashlib import md5
from itertools import chain
from itertools import islice
from string import Formatter
import io
import os
import posixpath
import subprocess
import threading

from config import Task, yaml_load
from utils import (ByrdException, LocalException, ObjectDict, RemoteException,
                   DummyClient, Env, spellcheck, spell, logger)

try:
    # This file is imported by setup.py at install time
    import keyring
    import paramiko
except ImportError:
    pass

__version__ = '0.0.2'
TAB = '\n    '



def get_secret(service, resource, resource_id=None):
    resource_id = resource_id or resource
    secret = keyring.get_password(service, resource_id)
    if not secret:
        secret = getpass('Password for %s: ' % resource)
        keyring.set_password(service, resource_id, secret)
    return secret


def get_passphrase(key_path):
    service = 'SSH private key'
    csum = md5(open(key_path, 'rb').read()).digest().hex()
    return get_secret(service, key_path, csum)


def get_password(host):
    service = 'SSH password'
    return get_secret(service, host)


def get_sudo_passwd():
    service = "Sudo password"
    return get_secret(service, 'sudo')


CONNECTION_CACHE = {}
def connect(host, auth):
    if host in CONNECTION_CACHE:
        return CONNECTION_CACHE[host]

    private_key_file = password = None
    if auth and auth.get('ssh_private_key'):
        private_key_file = os.path.expanduser(auth.ssh_private_key)
        if not os.path.exists(private_key_file):
            msg = 'Private key file "%s" not found' % private_key_file
            raise ByrdException(msg)
        password = get_passphrase(private_key_file)
    else:
        password = get_password(host)

    username, hostname = host.split('@', 1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password,
                   key_filename=private_key_file,
    )
    logger.debug(f'Connected to {hostname} as {username}')
    CONNECTION_CACHE[host] = client
    return client


def run_local(cmd, env, cli):
    # Run local task
    cmd = env.fmt(cmd)
    logger.info(env.fmt('{task_desc}', kind='new'))
    if cli.dry_run:
        logger.info('[dry-run] ' + cmd)
        return None
    logger.debug(TAB + TAB.join(cmd.splitlines()))
    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    stdout, stderr = process.communicate()
    success = process.returncode == 0
    if stdout:
        logger.debug(TAB + TAB.join(stdout.decode().splitlines()))
    if not success:
        raise LocalException(stdout, stderr)
    return ObjectDict(stdout=stdout, stderr=stderr)


def run_python(task, env, cli):
    # Execute a piece of python localy
    code = task.python
    logger.info(env.fmt('{task_desc}', kind='new'))
    if cli.dry_run:
        logger.info('[dry-run] ' + code)
        return None
    logger.debug(TAB + TAB.join(code.splitlines()))
    cmd = ['python', '-c', 'import sys;exec(sys.stdin.read())']
    if task.sudo:
        user = 'root' if task.sudo is True else task.sudo
        cmd = 'sudo -u {} -- {}'.format(user, cmd)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        env=env,
    )

    # Plug io
    out_buff = io.StringIO()
    err_buff = io.StringIO()
    log_stream(process.stdout, out_buff)
    log_stream(process.stderr, err_buff)
    process.stdin.write(code.encode())
    process.stdin.flush()
    process.stdin.close()
    success = process.wait() == 0
    process.stdout.close()
    process.stderr.close()
    out = out_buff.getvalue()
    if out:
        logger.debug(TAB + TAB.join(out.splitlines()))
    if not success:
        raise LocalException(out + err_buff.getvalue())
    return ObjectDict(stdout=out, stderr=err_buff.getvalue())


def log_stream(stream, buff):
    def _log():
        try:
            for chunk in iter(lambda: stream.readline(2048), ""):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode()
                buff.write(chunk)
        except ValueError:
            # read raises a ValueError on closed stream
            pass

    t = threading.Thread(target=_log)
    t.start()
    return t


def run_helper(client, cmd, env=None, in_buff=None, sudo=False):
    '''
    Helper function to run `cmd` command on remote host
    '''
    chan = client.get_transport().open_session()
    if env:
        chan.update_environment(env)

    stdin = chan.makefile('wb')
    stdout = chan.makefile('r')
    stderr = chan.makefile_stderr('r')
    out_buff = io.StringIO()
    err_buff = io.StringIO()
    out_thread = log_stream(stdout, out_buff)
    err_thread = log_stream(stderr, err_buff)

    if sudo:
        assert not in_buff, 'in_buff and sudo can not be combined'
        if isinstance(sudo, str):
            sudo_cmd = 'sudo -u %s -s' % sudo
        else:
            sudo_cmd = 'sudo -s'
        chan.exec_command(sudo_cmd)
        in_buff = cmd
    else:
        chan.exec_command(cmd)
    if in_buff:
        # XXX use a real buff (not a simple str) ?
        stdin.write(in_buff)
        stdin.flush()
        stdin.close()
        chan.shutdown_write()

    success = chan.recv_exit_status() == 0
    out_thread.join()
    err_thread.join()

    if not success:
        raise RemoteException(out_buff.getvalue() + err_buff.getvalue())

    res = ObjectDict(
        stdout = out_buff.getvalue(),
        stderr = err_buff.getvalue(),
    )
    return res


def run_remote(task, host, env, cli):
    res = None
    host = env.fmt(host)
    env.update({
        'host': extract_host(host),
    })
    if cli.dry_run:
        client = DummyClient()
    else:
        client = connect(host, cli.cfg.auth)

    if task.run:
        cmd = env.fmt(task.run)
        prefix = ''
        if task.sudo:
            if task.sudo is True:
                prefix = '[sudo] '
            else:
                 prefix = '[sudo as %s] ' % task.sudo
        msg = prefix + '{host}: {task_desc}'
        logger.info(env.fmt(msg, kind='new'))
        logger.debug(TAB + TAB.join(cmd.splitlines()))
        if cli.dry_run:
            logger.info('[dry-run] ' + cmd)
        else:
            res = run_helper(client, cmd, env=env, sudo=task.sudo)

    elif task.send:
        local_path = env.fmt(task.send)
        if not os.path.exists(local_path):
            raise ByrdException('Path "%s" not found'  % local_path)
        else:
            send(client, env, cli, task)

    else:
        raise ByrdException('Unable to run task "%s"' % task.name)

    if res and res.stdout:
        logger.debug(TAB + TAB.join(res.stdout.splitlines()))

    return res

def send(client, env, cli, task):
    fmt = task.fmt and Env(env, {'fmt': 'new'}).fmt(task.fmt) or None
    local_path = env.fmt(task.send)
    remote_path = env.fmt(task.to)
    dry_run = cli.dry_run
    with client.open_sftp() as sftp:
        if os.path.isfile(local_path):
            send_file(sftp, os.path.abspath(local_path), remote_path, env,
                      dry_run=dry_run, fmt=fmt)
        elif os.path.isdir(local_path):
            for root, subdirs, files in os.walk(local_path):
                rel_dir = os.path.relpath(root, local_path)
                rel_dirs = os.path.split(rel_dir)
                rem_dir = posixpath.join(remote_path, *rel_dirs)
                run_helper(client, 'mkdir -p {}'.format(rem_dir))
                for f in files:
                    rel_f = os.path.join(root, f)
                    rem_file = posixpath.join(rem_dir, f)
                    send_file(sftp, os.path.abspath(rel_f), rem_file, env,
                              dry_run=dry_run, fmt=fmt)
        else:
            msg = 'Unexpected path "%s" (not a file, not a directory)'
            raise ByrdException(msg % local_path)


def send_file(sftp, local_path, remote_path, env, dry_run=False, fmt=None):
    if not fmt:
        logger.info(f'[send] {local_path} -> {remote_path}')
        lines = islice(open(local_path), 30)
        logger.debug('File head:' + TAB.join(lines))
        if not dry_run:
            sftp.put(local_path, remote_path)
        return
    # Format file content and save it on remote
    local_relpath = os.path.relpath(local_path)
    logger.info(f'[fmt] {local_relpath} -> {remote_path}')
    content = env.fmt(open(local_path).read(), kind=fmt)
    lines = islice(content.splitlines(), 30)
    logger.debug('File head:' + TAB.join(lines))
    if not dry_run:
        fh = sftp.open(remote_path, mode='w')
        fh.write(content)
        fh.close()


def run_task(task, host, cli, env=None):
    '''
    Execute one task on one host (or locally)
    '''
    if task.local:
        res = run_local(task.local, env, cli)
    elif task.python:
        res = run_python(task, env, cli)
    else:
        res = run_remote(task, host, env, cli)

    if task.get('assert'):
        eval_env = {
            'stdout': res.stdout.strip(),
            'stderr': res.stderr.strip(),
        }
        assert_ = env.fmt(task['assert'])
        ok = eval(assert_, eval_env)
        if ok:
            logger.info('Assert ok')
        else:
            raise ByrdException('Assert "%s" failed!' % assert_)

    if task.get('warn'):
        msg = Env(env, res).fmt(task['warn'])
        logger.warning(msg)
    return res


def run_batch(task, hosts, cli, global_env=None):
    '''
    Run one task on a list of hosts
    '''
    out = None
    export_env = {}
    task_env = global_env.fmt(task.get('env', {}))
    if not hosts and task.networks:
        hosts = list(chain(*(spellcheck(cli.cfg.networks, n).hosts
                             for n in task.networks)))

    if task.get('multi'):
        parent_env = Env(export_env, task_env, global_env)
        parent_sudo = task.sudo
        for pos, step in enumerate(task.multi):
            task_name = step.task
            if task_name:
                # _cfg contain "local" config wrt the task
                siblings = task._cfg.tasks
                sub_task = spellcheck(siblings, task_name)
                sudo = step.sudo or sub_task.sudo or parent_sudo
            else:
                # reify a task out of attributes
                sub_task = Task.parse(step)
                sub_task._path = '%s->[%s]' % (task._path, pos)
                sudo = sub_task.sudo or parent_sudo

            sub_task.sudo = sudo
            network = step.get('network')
            if network:
                hosts = spellcheck(cli.cfg.networks, network).hosts
            child_env = step.get('env', {})
            child_env = parent_env.fmt(child_env)
            out = run_batch(sub_task, hosts, cli, Env(child_env, parent_env))
            out = out.decode() if isinstance(out, bytes) else out
            export_env['_'] = out
            if step.export:
                export_env[step.export] = out

    else:
        task_env.update({
            'task_desc': global_env.fmt(task.desc),
            'task_name': task.name,
        })
        parent_env = Env(task_env, global_env)
        if task.get('fmt'):
            parent_env.fmt_kind = task.fmt

        res = None
        if task.once and (task.local or task.python):
            res = run_task(task, None, cli, parent_env)
        elif hosts:
            for host in hosts:
                env_host = extract_host(host)
                parent_env.update({
                    'host': env_host,
                })
                res = run_task(task, host, cli, parent_env)
                if task.once:
                    break
        else:
            logger.warning('Nothing to do for task "%s"' % task._path)
        out = res and res.stdout.strip() or ''
    return out


def extract_host(host_string):
    return host_string and host_string.split('@')[-1] or ''




def get_hosts_and_tasks(cli, cfg):
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
            raise ByrdException (msg)

    # Collect custom tasks from cli
    customs = []
    for cli_key in ('run', 'run_local', 'run_python'):
        cmd_key = cli_key.rsplit('_', 1)[-1]
        customs.extend('%s: %s' % (cmd_key, ck) for ck in cli[cli_key])
    for custom_task in customs:
        task = Task.parse(yaml_load(custom_task))
        task.desc = 'Custom command'
        tasks.append(task)

    hosts = list(chain.from_iterable(n.hosts for n in networks))

    return dict(hosts=hosts, tasks=tasks)


def info(cli):
    formatter = Formatter()
    for name, attr in cli.cfg.tasks.items():
        kind = 'remote'
        if attr.python:
            kind = 'python'
        elif attr.local:
            kind = 'local'
        elif attr.multi:
            kind = 'multi'
        elif attr.send:
            kind = 'send file'

        print(f'{name} [{kind}]:\n\tDescription: {attr.desc}')

        values = []
        for v in attr.values():
            if isinstance(v, list):
                values.extend(v)
            elif isinstance(v, dict):
                values.extend(v.values())
            else:
                values.append(v)
        values = filter(lambda x: isinstance(x, str), values)
        fmt_fields = [i[1] for v in values for i in formatter.parse(v) if i[1]]
        if fmt_fields:
            variables = ', '.join(sorted(set(fmt_fields)))
        else:
            variables = None

        if variables:
            print(f'\tVariables: {variables}')
