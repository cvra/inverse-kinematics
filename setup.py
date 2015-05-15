#!/usr/bin/env python

from setuptools import setup, find_packages

args = dict(
    name='pickit',
    version='0.2',
    description='Kinematics and motion planning for CVRA arm manipulators',
    packages=['pickit'],
    install_requires=['numpy'],
    author='Salah-Eddine Missri',
    author_email='missrisalaheddine@gmail.com',
    url='https://github.com/cvra',
    license='BSD'
)

setup(**args)
