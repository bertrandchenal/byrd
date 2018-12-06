# Byrd

Byrd is a simple deployment tool based on paramiko. The config file
format is inspired by [Sup](https://github.com/pressly/sup) (but not
identical). In contrast to sup, Byrd is meant to be invoked from any
OS (aka Windows support). This project is in alpha stage, please
handle carefully.

The name Byrd is a reference to
[Donald Byrd](https://en.wikipedia.org/wiki/Donald_Byrd).


# Quickstart

## Basic Example

By default byrd will use `bd.yaml` as config file:

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
$ bd time
INFO:2018-08-01 23:14:05: Load config bd.yaml
$ bd  time -v
INFO:2018-08-01 23:14:21: Load config bd.yaml
INFO:2018-08-01 23:14:21: Print current time (on local machine)
2018-08-01T23:14:21+02:00
$ bd health web -v
INFO:2018-08-01 23:14:25: Load config bd.yaml
INFO:2018-08-01 23:14:25: web1.example.com: Get basic health info
 23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
INFO:2018-08-01 23:14:26: web2.example.com: Get basic health info
  23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
```


You can also pass `--dry-run` (or `-d`) to print what would have been done:
```
$ bd web health --dry-run
INFO:2018-08-13 14:36:05: Load config bd.yaml
INFO:2018-08-13 14:36:05: web1.example.com: Get basic health info
INFO:2018-08-13 14:36:05: [DRY-RUN] uptime
INFO:2018-08-13 14:36:05: web2.example.com: Get basic health info
INFO:2018-08-13 14:36:05: [DRY-RUN] uptime
```


## Environment

Each command is auto-formatted based on current environment. When Byrd
is invoked, the basic environment is initialized with the one from
parent process (literally `os.environ`). It is then augmented by the
top level `env` directive of the config file and then command-line
`-e` arguments are added.

The command formatting use the string formatting tool from python, see
[Python documentation](https://docs.python.org/3.4/library/string.html#format-examples)
for examples.  In particular, you have to use a brace character (`{`
or `}`) in a command, you can double it (`{{` or `}}`) to escape
formatting.

Each task can also declare extra key-value pairs in the `env`
directive, and those are kept when sub-tasks are ran. Siblings tasks
are not impacted. Those values are also formatted, this allows you to
rename environment variables (especially useful when running a
sub-task that expect a variable to be set up)


## Multi-tasks

The following example shows how Byrd handles environment variables,
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
        export: my_var  # tells byrd to use task ouput to set my_var
      - task: echo-var
```

We can then do the following:

```
$ bd both -v
INFO:2018-08-01 23:00:37: Load config bd.yaml
INFO:2018-08-01 23:00:37: Simple echo 
ECHO!
INFO:2018-08-01 23:00:37: Echo an env variable
ECHO!
$ bd both -e what="WHAT?" -v
INFO:2018-08-01 23:01:15: Load config bd.yaml
INFO:2018-08-01 23:01:15: Simple echo
WHAT?
INFO:2018-08-01 23:01:15: Echo an env variable
WHAT?
```

As you can see the `export` directive tells Byrd to save the result
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

Environment is used to format command, but it is also used to define
the initial environment of command (duh!), so you can access it with
the os module (for example `print(os.environ['MY_VAR']`). It works for
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

Currently, Byrd only supports private key authentication. You can add
an `auth` section that tells where to find your private key:

```
auth:
  ssh_private_key: path/to/deploy_key_rsa
```

On the following invocation, Byrd will ask your passphrase for the
key. This passphrase will be saved in your OS keyring (thanks to
[the keyring module](https://github.com/jaraco/keyring).)


## Load other config files

The top-level directive `load` allows to import other config files and
merge them with the current one, so with `bd.yaml` containing:

```
load:
  - file: network.yaml
    as: net
tasks:
  echo-host:
    local: echo {host}
```

and `network.yaml` containing:

```
networks:
  web:
    hosts:
      - web1.example.com
      - web2.example.com
```

We can now run:

```
$ bd net/web echo-host --dry-run -v
INFO:2018-08-14 11:27:33: Load config bd.yaml
INFO:2018-08-14 11:27:33: Load config network.yaml
INFO:2018-08-14 11:27:33: echo-host
INFO:2018-08-14 11:27:33: [DRY-RUN] echo web1.example.com
INFO:2018-08-14 11:27:33: echo-host
INFO:2018-08-14 11:27:33: [DRY-RUN] echo web2.example.com
```


# Roadmap

- Add doc about load directive (and namespacing)
- Add a contrib directory with ready-made tasks for common
  operations
- Add password-based auth, add per-network auth.
