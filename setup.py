from distutils.core import setup
import setuptools

try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = None


setup(
    name='tf2deeplab',
    version='0.0.0',
    license='MIT',
    author='https://github.com/srihari-humbarwadi',
    packages=setuptools.find_packages(),
    description='DeepLabV3 implementation in TF2.0',
    include_package_data=True,
    python_requires='>=3.6.5',
    install_requires=requirements,
)
