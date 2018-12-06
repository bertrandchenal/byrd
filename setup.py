#!/usr/bin/env python
from setuptools import setup
from glob import glob
import os

import byrd

long_description = '''

Byrd is yet another deployment tool. Byrd is a mashup of Paramiko
(https://www.paramiko.org/) and the sup config file layout
(https://github.com/pressly/sup).

The name Byrd is a reference to Donald Byrd.
'''

description = ('Simple deployment tool based on Paramiko')
basedir, _ = os.path.split(__file__)
pkg_yaml = glob(os.path.join(basedir, 'pkg', '*yaml'))

setup(name='Byrd',
      version=byrd.__version__,
      description=description,
      long_description=long_description,
      author='Bertrand Chenal',
      author_email='bertrand@adimian.com',
      url='https://bitbucket.org/bertrandchenal/byrd',
      license='MIT',
      py_modules=['byrd'],
      entry_points={
          'console_scripts': [
              'bd = byrd:main',
          ],
      },
      data_files=[('pkg', pkg_yaml)],
      install_requires=[
          'paramiko',
          'pyyaml',
          'keyring',
      ],
)
