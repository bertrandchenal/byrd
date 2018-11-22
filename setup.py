#!/usr/bin/env python
from setuptools import setup
from glob import glob
import baker

long_description = '''

Baker is yet another deployment tool. Baker is a mashup of Paramiko
(https://www.paramiko.org/) and the sup config file layout
(https://github.com/pressly/sup).

The name Baker is a reference to Chet Baker.
'''

description = ('Simple deployment tool based on Paramiko')

pkg_yaml = glob('pkg/*.yaml')

setup(name='Baker',
      version=baker.__version__,
      description=description,
      long_description=long_description,
      author='Bertrand Chenal',
      author_email='bertrand@adimian.com',
      url='https://bitbucket.org/bertrandchenal/baker',
      license='MIT',
      py_modules=['baker'],
      entry_points={
          'console_scripts': [
              'bk = baker:main',
          ],
      },
      data_files=[('pkg', pkg_yaml)],
)
