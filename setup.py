from setuptools import setup
from setuptools import find_packages

package_data = dict(pyrot=['*.py',
                           'pyrot/*.py',
                           ])

if __name__ == "__main__":
      setup(name='pyrot',
            version='0.0.0',
            author='Dominik Lentrodt',
            author_email='dominik@lentrodt.com',
            license='GPLv3',
            license_files = ('LICENSE.txt',),
            description = 'A python package for the physics of 1D Fabry-Perot cavities interacting with atoms.',
            url = 'https://github.com/dlentrodt/pyrot',
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            package_data=package_data
          )