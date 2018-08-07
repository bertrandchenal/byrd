import glob

from baker import load


def pytest_generate_tests(metafunc):
    if 'cfg' in metafunc.fixturenames:
        configs = []
        for name in glob.glob('tests/*yaml'):
            cfg = load(name)
            configs.append(cfg)
        metafunc.parametrize("cfg", configs)
