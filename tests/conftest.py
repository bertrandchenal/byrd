import glob
import io
import logging

import pytest

from baker import load, logger, log_handler

# Disable default handler
logger.removeHandler(log_handler)


@pytest.yield_fixture(scope='function')
def log_handler(request):
    buff = io.StringIO()
    handler = logging.StreamHandler(buff)
    logger.addHandler(handler)
    yield buff
    logger.removeHandler(handler)


def pytest_generate_tests(metafunc):
    if 'cfg' in metafunc.fixturenames:
        configs = []
        logs = []
        for name in glob.glob('tests/*yaml'):
            print(name)
            configs.append(load(name))
            log_file = name.replace('.yaml', '.log')
            logs.append(open(log_file).read())
        metafunc.parametrize("cfg,nominal_log", zip(configs, logs))
        # metafunc.parametrize("log", logs)
