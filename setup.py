from setuptools import setup

setup(name='botorch_generation',
      version='0.0.1',
      description="Company made package for BO generation using Botorch",
      license='See license',
      packages=['models', "utils", "search", "database", "chemicals"],
      zip_safe=False,
      include_package_data=True)