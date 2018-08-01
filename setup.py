#!/usr/bin/env python
from setuptools import setup

import baker

long_description = '''
Baker is yet another deployment tool. Baker is a mashup of Fabric
(http://www.fabfile.org/) and the sup config file layout
(https://github.com/pressly/sup).

The name Baker is a reference to Chet Baker.
'''

description = ('Simple deployment tool based on Fabric')

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
  )
