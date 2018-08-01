# Baker

Baker is a simple deployment tool based on Fabric. The config file
format is inspired (but not identical) by
[Sup](https://github.com/pressly/sup), but contrary to sup Baker is
meant to be invoked from any os.

## Basic Example

By default baker will use `bk.yaml` as config file:

```yaml
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


Based on the above file, one can run the following operations:

```bash
$ bk time
INFO:2018-08-01 22:35:48: Load config bk.yaml
INFO:2018-08-01 22:35:48: RUN time locally
2018-08-01T22:45:33+02:00

$ bk health web
INFO:2018-08-01 22:40:39: Load config bk.yaml
INFO:2018-08-01 22:40:39: RUN health ON web1.example.com
 22:40:39 up 7 days,  5:55,  4 users,  load average: 0,33, 0,28, 0,31
INFO:2018-08-01 22:40:39: RUN health ON web2.example.com
  22:40:40 up 7 days,  5:55,  4 users,  load average: 0,30, 0,27, 0,30
```


## Multi-tasks

The following example shows how Baker handles environment variables,
and how to assemble tasks.

```
tasks:
  echo-one:
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
      - task: echo-one
        export: my_var
      - task: echo-two
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
