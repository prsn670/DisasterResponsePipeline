from setuptools import setup

setup(name='ETL',
      version='0.1',
      description='Extract, Transform, Load',
      packages=['etl'],
      zip_safe=False, install_requires=['pandas', 'sqlalchemy'])