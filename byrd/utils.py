import sys
from itertools import chain
from collections import defaultdict, ChainMap
from contextlib import contextmanager
import logging

log_fmt = '%(levelname)s:%(asctime).19s: %(message)s'
logger = logging.getLogger('byrd')
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter(log_fmt))
logger.addHandler(log_handler)


class ByrdException(Exception):
    pass

class FmtException(ByrdException):
    pass

class ExecutionException(ByrdException):
    pass

class RemoteException(ExecutionException):
    pass

class LocalException(ExecutionException):
    pass

def enable_logging_color():
    try:
        import colorama
    except ImportError:
        return

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
    logger.removeHandler(log_handler)
    handler = Handler()
    handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(handler)
    logger.propagate = 0



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
        return objdict[word]

    candidates = objdict.get('_candidates')
    if not candidates:
        candidates = gen_candidates(list(objdict))
        objdict._candidates = candidates

    msg = '"%s" not found in %s' % (word, objdict._path)
    matches = spell(candidates, word)
    if matches:
        msg += ', try: %s' % ' or '.join(matches)
    raise ByrdException(msg)


def abort(msg):
    logger.error(msg)
    sys.exit(1)


class ObjectDict(dict):
    """
    Simple objet sub-class that allows to transform a dict into an
    object, like: `ObjectDict({'ham': 'spam'}).ham == 'spam'`
    """

    # Meta allows to hide all the keys starting with an '_'
    _meta = {}

    def copy(self):
        res = ObjectDict(super().copy())
        ObjectDict._meta[id(res)] = ObjectDict._meta.get(id(self), {}).copy()
        return res

    def __getattr__(self, key):
        if key.startswith('_'):
            return ObjectDict._meta[id(self), key]

        if key in self:
            return self[key]
        else:
            return None

    def __setattr__(self, key, value):
        if key.startswith('_'):
            ObjectDict._meta[id(self), key] = value
        else:
            self[key] = value


class DummyClient:
    '''
    Dummy Paramiko client, mainly usefull for testing & dry runs
    '''

    @contextmanager
    def open_sftp(self):
        yield None

class Env(ChainMap):

    def __init__(self, *dicts):
        self.fmt_kind = 'new'
        return super().__init__(*filter(lambda x: x is not None, dicts))

    def fmt_env(self, child_env, kind=None):
        new_env = {}
        for key, val in child_env.items():
            # env wrap-around!
            new_val = self.fmt(val, kind=kind)
            if new_val == val:
                continue
            new_env[key] = new_val
        return Env(new_env, child_env)

    def fmt_string(self, string, kind=None):
        fmt_kind = kind or self.fmt_kind
        try:
            if fmt_kind == 'old':
                return string % self
            else:
                return string.format(**self)
        except KeyError as exc:
            msg = 'Unable to format "%s" (missing: "%s")'% (string, exc.args[0])
            candidates = gen_candidates(self.keys())
            key = exc.args[0]
            matches = spell(candidates, key)
            if matches:
                msg += ', try: %s' % ' or '.join(matches)
            raise FmtException(msg )
        except IndexError:
            msg = 'Unable to format "%s", positional argument not supported'
            raise FmtException(msg)

    def fmt(self, what, kind=None):
        if isinstance(what, str):
            return self.fmt_string(what, kind=kind)
        return self.fmt_env(what, kind=kind)
