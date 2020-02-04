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

from setuptools import setup, Extension

from Cython.Build import cythonize

NAME = 'n2'
VERSION = '0.1.5'


def long_description():
    with io.open('README.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


def define_extensions(**kwargs):
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
