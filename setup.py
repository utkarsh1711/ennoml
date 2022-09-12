from setuptools import setup

setup(name='ennoml',
    version='0.1',
    description='test package to run on qbee.io',
    author='Utkarsh',
    author_email='author@somemail.com',
    license='MIT',
    packages=['.ennoml','.ennoml.data','.ennoml.testing','.ennoml.training','.ennoml.util','.ennoml.models'],
    install_requires=['tensorflow-gpu>=2.3,<2.7','pandas','Pillow','opencv-python','numpy', 'scikit-image'])
    # zip_safe=False)
