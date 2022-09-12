from setuptools import setup

setup(name='ennoml',
    version='0.1',
    description='test package to run on qbee.io',
    author='Utkarsh',
    author_email='author@somemail.com',
    license='MIT',
    packages=['.ennoml','.ennoml.data','.ennoml.testing','.ennoml.training','.ennoml.util','.ennoml.models'],
#     scripts=['bin/UnitTest.py'],
    zip_safe=False)
