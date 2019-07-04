from collections import OrderedDict

from .utils import (ByrdException, ObjectDict, spell,
                    gen_candidates)


try:
    import yaml
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


class Node:

    @staticmethod
    def fail(path, kind):
        msg = 'Error while parsing config: expecting "%s" while parsing "%s"'
        raise ByrdException(msg % (kind, '->'.join(path)))

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
                        if path:
                            msg = 'Attribute "%s" not understood in %s' % (
                                key, path)
                        else:
                            msg = 'Top-level attribute "%s" not understood' % (
                                key)
                        candidates = gen_candidates(children.keys())
                        matches = spell(candidates, key)
                        if matches:
                            msg += ', try: %s' % ' or '.join(matches)
                        raise ByrdException(msg)

                for name, child_class in children.items():
                    if name not in cfg:
                        continue
                    res[name] = child_class.parse(cfg[name], path + (name,))

        elif type_name == 'list':
            if not isinstance(cfg, list):
                cls.fail(path, type_name)
            child_class = children[0]
            res = [child_class.parse(c, path+ ('[%s]' % pos,))
                   for pos, c in enumerate(cfg)]

        else:
            if not isinstance(cfg, cls._type):
                cls.fail(path, type_name)
            res = cfg

        return cls.setup(res, path)

    @classmethod
    def setup(cls, values, path):
        if isinstance(values, ObjectDict):
            values._path = '->'.join(path)
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
        'network': Atom,
    }

class MultiList(Node):
    _children = [Multi]

class Task(Node):
    _children = {
        'desc': Atom,
        'local': Atom,
        'python': Atom,
        'once': Atom,
        'run': Atom,
        'sudo': Atom,
        'send': Atom,
        'to': Atom,
        'assert': Atom,
        'warn': Atom,
        'env': EnvNode,
        'multi': MultiList,
        'fmt': Atom,
        'networks': AtomList,
        # TODO add support for auth here
    }

    @classmethod
    def setup(cls, values, path):
        values['name'] = path and path[-1] or ''
        if 'desc' not in values:
            values['desc'] = values.get('name', '')
        super().setup(values, path)
        return values

# Multi can also accept any task attribute:
Multi._children.update(Task._children)


class TaskGroup(Node):
    _children = {
        '*': Task,
    }

class LoadNode(Node):
    _children = {
        'file': Atom,
        'pkg': Atom,
        'as': Atom,
    }

class LoadList(Node):
    _children = [LoadNode]

class ConfigRoot(Node):
    _children = {
        'networks': Network,
        'tasks': TaskGroup,
        'auth': Auth,
        'env': EnvNode,
        'load': LoadList,
    }
