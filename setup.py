"""
Copyright (C) 2020-2021 Dominik Lentrodt

This file is part of pygreenfn.

pygreenfn is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pygreenfn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pygreenfn.  If not, see <http://www.gnu.org/licenses/>.
"""


from setuptools import setup

package_data = dict(pygreenfn=['*.py',
                               'pygreenfn/*.py',
                               'pygreenfn_pol/*.py',
                               ])

setup(name='pygreenfn',
      version='0.0.0',
      author='Dominik Lentrodt',
      author_email='dominik.lentrodt@mpi-hd.mpg.de',
      packages=['pygreenfn', 'pygreenfn_pol'],
      package_data=package_data
    )