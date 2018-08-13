# Baker

Baker is a simple deployment tool based on paramiko. The config file
format is inspired by [Sup](https://github.com/pressly/sup) (but not
identical). In contrast to sup, Baker is meant to be invoked from any
OS (aka Windows support). This project is in alpha stage, please
handle carefully.

The name Baker is a reference to
[Chet Baker](https://en.wikipedia.org/wiki/Chet_Baker).


# Quickstart

## Basic Example

By default baker will use `bk.yaml` as config file:

```
networks:
  web:
    hosts:
      - web1.example.com
      - web2.example.com
  db:
    hosts:
      - db1.example.com
      - db2.example.com

tasks:
  health:
    desc: Get basic health info
    run: uptime

  time:
    desc: Print current time (on local machine)
    local: date -Iseconds
    once: true
```


Based on the above file, one can run the following operations (imagine
that the INFO lines are colored):

```
$ bk time
INFO:2018-08-01 23:14:05: Load config bk.yaml
$ bk  time
INFO:2018-08-01 23:14:21: Load config bk.yaml
INFO:2018-08-01 23:14:21: Print current time (on local machine)
2018-08-01T23:14:21+02:00
$ bk health web
INFO:2018-08-01 23:14:25: Load config bk.yaml
INFO:2018-08-01 23:14:25: web1.example.com: Get basic health info
 23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
INFO:2018-08-01 23:14:26: web2.example.com: Get basic health info
  23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
```


You can also pass `--dry-run` (or `-d`) to print what would have been done:
```
$ bk web health --dry-run
INFO:2018-08-13 14:36:05: Load config bk.yaml
INFO:2018-08-13 14:36:05: web1.example.com: Get basic health info
INFO:2018-08-13 14:36:05: [DRY-RUN] uptime
INFO:2018-08-13 14:36:05: web2.example.com: Get basic health info
INFO:2018-08-13 14:36:05: [DRY-RUN] uptime
```


## Multi-tasks

The following example shows how Baker handles environment variables,
and how to assemble tasks.

```
tasks:
  echo:
    desc: Simple echo
    local: echo "{what}"
    once: true
    env:
      what: "ECHO!"

  echo-var:
    desc: Echo an env variable
    local: echo {my_var}
    once: true
    
  both:
    desc: Run both tasks
    multi:
      - task: echo
        export: my_var  # tells baker to use task ouput to set my_var
      - task: echo-var
```

We can then do the following:

```
$ bk both
INFO:2018-08-01 23:00:37: Load config bk.yaml
INFO:2018-08-01 23:00:37: Simple echo 
ECHO!
INFO:2018-08-01 23:00:37: Echo an env variable
ECHO!
$ bk both -e what="WHAT?"
INFO:2018-08-01 23:01:15: Load config bk.yaml
INFO:2018-08-01 23:01:15: Simple echo
WHAT?
INFO:2018-08-01 23:01:15: Echo an env variable
WHAT?
```

As you can see the `export` directive tells Baker to save the result
of a command under a given environment variable.


## Python

Python code can be added with a python directive:

```
  hello-python:
    desc: Says hello with python
    python: |
      for i in range(10):
          print('hello')
    once: true
```

Environement is used to format command, but it is also used to define
the initial environement of command (duh!), so you can access it with
the os module (for example �print(os.envoron['MY_VAR']`). It works for
python directive but also for local and remote commands.

## Assert

```
  hello-python:
    desc: Says hello with python
    python: |
      for i in range(10):
          print('hello')
    once: true
    assert: "len(stdout.splitlines()) == 10"

```

You can also add `assert` directives, they are run as a python eval
within the current env, and can access two extra variables: `stdout`
and `stderr`.


## SSH Authentication

Currently, Baker only supports private key authentication. You can add
an `auth` section that tells where to find your private key:

```
auth:
  ssh_private_key: path/to/deploy_key_rsa
```

On the following invocation, Baker will ask your passphrase for the
key. This passphrase will be saved in your OS keyring (thanks to
[the keyring module](https://github.com/jaraco/keyring).)


# Roadmap

- Implement: add a contrib directory with ready-made tasks for common
  operations, add password-based auth, add per-network auth.
