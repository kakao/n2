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

import io
import os
import glob
import platform
import subprocess

from Cython.Build import cythonize
from setuptools import Extension, setup

NAME = 'n2'
VERSION = '0.1.6'


def long_description():
    with io.open('README.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


def set_binary_mac():
    gcc_dir = subprocess.check_output("brew --prefix gcc", shell=True).decode().strip()
    gcc_dir = os.path.join(gcc_dir, 'bin')
    gpp_binaries = glob.glob(os.path.join(gcc_dir, 'g++-[0-9]'))
    gcc_binaries = glob.glob(os.path.join(gcc_dir, 'gcc-[0-9]'))
    binaries = [gcc_binaries, gpp_binaries]
    targets = ["CC", "CXX"]
    fail = False
    for binary, target in zip(binaries, targets):
        if binary:
            binary = sorted(binary, key=lambda x: int(x.split('-')[1]))[-1]
            os.environ[target] = os.path.join(gcc_dir, binary)
        else:
            fail = True
            break

    if fail:
        raise AttributeError('No GCC available. Install gcc from Homebrew using brew install gcc.')


def define_extensions(**kwargs):
    system = platform.system().lower()
    if "windows" in system:  # Windows
        raise AttributeError("Installation on Windows is not supported yet.")
    elif "darwin" in system:  # osx
        set_binary_mac()

    libraries = []
    extra_link_args = []
    extra_compile_args = ['-std=c++14', '-O3', '-fPIC', '-march=native']
    extra_link_args.append('-fopenmp')
    extra_compile_args.append('-fopenmp')

    sources = ['./src/heuristic.cc', './src/hnsw.cc', './src/hnsw_node.cc',
               './src/hnsw_build.cc', './src/hnsw_model.cc', './src/hnsw_search.cc',
               './src/mmap.cc', './bindings/python/n2.pyx']

    boost_dirs = ['assert', 'bind', 'concept_check', 'config', 'core', 'detail', 'heap', 'iterator', 'mp11', 'mpl',
                  'parameter', 'preprocessor', 'static_assert', 'throw_exception', 'type_traits', 'utility']
    include_dirs = ['./include/', './third_party/spdlog/include/', './third_party/eigen']
    include_dirs.extend(['third_party/boost/' + b + '/include/' for b in boost_dirs])

    client_ext = Extension(name='n2',
                           sources=sources,
                           extra_compile_args=extra_compile_args,
                           libraries=libraries,
                           extra_link_args=extra_link_args,
                           include_dirs=include_dirs,
                           language="c++",)
    return cythonize(client_ext)


setup(
    name=NAME,
    version=VERSION,
    description='Approximate Nearest Neighbor library',
    long_description=long_description(),
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
