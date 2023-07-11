#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import setuptools


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="mono_depth_estimation_aicrowd",
    version="0.0.1",
    packages=setuptools.find_packages(exclude=["models"]),
    install_requires=requirements,
)
