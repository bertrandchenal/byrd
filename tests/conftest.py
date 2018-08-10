import glob
import io
import logging

import pytest

from baker import logger, log_handler, yaml_load, ObjectDict

# Disable default handler
logger.removeHandler(log_handler)


@pytest.yield_fixture(scope='function')
def log_buff(request):
    buff = io.StringIO()
    handler = logging.StreamHandler(buff)
    handler.setLevel('DEBUG')
    logger.setLevel('DEBUG')
    logger.addHandler(handler)
    yield buff
    logger.removeHandler(handler)


def pytest_addoption(parser):
    parser.addoption("-F", "--filter", help="Filter yaml files")


def pytest_generate_tests(metafunc):
    if not 'test_cfg' in metafunc.fixturenames:
        return

    test_configs = []
    ids = []
    fltr = metafunc.config.option.filter
    for name in glob.glob('tests/*test.yaml'):
        if fltr and not fltr in name:
            continue
        test_cfg = ObjectDict(yaml_load(open(name)))
        ids.append(test_cfg.cfg)
        test_configs.append(test_cfg)
    metafunc.parametrize("test_cfg", test_configs, ids=ids)
