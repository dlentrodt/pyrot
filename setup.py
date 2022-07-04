from setuptools import setup

package_data = dict(pyrot=['*.py',
                           'pyrot/*.py',
                           ])

if __name__ == "__main__":
      setup()

setup(name='pyrot',
      version='0.0.0',
      author='Dominik Lentrodt',
      author_email='dominik@lentrodt.com',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      package_data=package_data
    )