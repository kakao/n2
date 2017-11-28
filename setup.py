# Copyright 2017 Kakao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import platform
import subprocess

from setuptools import setup, Extension

from Cython.Build import cythonize

NAME = 'n2'
VERSION = '0.1.3'

use_openmp = True


def define_extensions():
    global use_openmp
    libraries = []
    extra_link_args = []
    extra_compile_args = ['-std=c++11', '-O3', '-fPIC', '-march=native']
    if use_openmp:
        extra_link_args.append('-fopenmp')
        extra_compile_args.append('-fopenmp')

    sources = ['./src/base.cc', './src/distance.cc', './src/heuristic.cc',  
        './src/hnsw.cc', './src/hnsw_node.cc',  './src/mmap.cc', 
        './bindings/python/n2.pyx']

    client_ext = Extension(name='n2',
                           sources=sources,
                           extra_compile_args=extra_compile_args,
                           libraries=libraries,
                           extra_link_args=extra_link_args,
                           include_dirs=['./include/', './third_party/spdlog/include/'],
                           language="c++",)
    return cythonize(client_ext)


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support.
    """
    # For macports and homebrew
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']

    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()

setup(
    name=NAME,
    version=VERSION,
    description='Approximate Nearest Neighbor library',
    author='Kakao.corp',
    author_email='recotech.kakao@gmail.com',
    license='Apache License 2.0',
    install_requires=[
        "cython",
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Topic :: Software Development :: Libraries :: Python Modules'],

    keywords='Approximate Nearest Neighbor',
    ext_modules=define_extensions(),
)
