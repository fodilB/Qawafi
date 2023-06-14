import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name='poetry_diacritizer',
      version='0.0.1',
      url='https://github.com/zaidalyafeai/Arabic_Diacritization',
      discription="Arabic poetry diacritizer",
      long_description=readme,
      long_description_content_type='text/markdown',
      author_email='arabicmachinelearning@gmail.com',
      license='MIT',
      packages=['poetry_diacritizer'],
      install_requires=required,
      python_requires=">=3.9",
      include_package_data=True,
      zip_safe=False,
      )
