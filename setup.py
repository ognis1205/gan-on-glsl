from setuptools import setup, find_packages
from pkg_resources import parse_requirements


setup(
    name="cppn",
    version="0.1.0",
    description="CPPN: Compositional Pattern Producing Networ.",
    author="Shingo OKAWA",
    python_requires="==3.7.*",
    install_requires=[
        'tensorflow',
        'Pillow',
        'matplotlib'
    ],
    packages=find_packages(exclude=["test", "test.*"]))
