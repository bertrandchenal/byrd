# Baker

Baker is a simple deployment tool based on Fabric. The config file
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
$ bk -c examples/bk1.yaml
INFO:2018-08-01 23:14:05: Load config examples/bk1.yaml
[23:14:05] bch@aldebaran:~/dev/baker$ bk -c examples/bk1.yaml time
INFO:2018-08-01 23:14:21: Load config examples/bk1.yaml
INFO:2018-08-01 23:14:21: RUN time locally
2018-08-01T23:14:21+02:00
$ bk -c examples/bk1.yaml health web
INFO:2018-08-01 23:14:25: Load config examples/bk1.yaml
INFO:2018-08-01 23:14:25: RUN health ON web1.example.com
 23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
INFO:2018-08-01 23:14:26: RUN health ON web2.example.com
  23:14:26 up 7 days,  6:28,  4 users,  load average: 0,30, 0,26, 0,22
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
$ bk -c examples/bk2.yaml both                
INFO:2018-08-01 23:00:37: Load config examples/bk2.yaml
INFO:2018-08-01 23:00:37: RUN echo locally
ECHO!
INFO:2018-08-01 23:00:37: RUN echo-var locally
ECHO!
$ bk -c examples/bk2.yaml both -e what="WHAT?"
INFO:2018-08-01 23:01:15: Load config examples/bk2.yaml
INFO:2018-08-01 23:01:15: RUN echo locally
WHAT?
INFO:2018-08-01 23:01:15: RUN echo-var locally
WHAT?
```

# Roadmap

- Document: assert, send dry-run and auth
- Add tests
- Implement: import of other config file, add a contrib directory with
  ready-made tasks for common operations
