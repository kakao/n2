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
import sys
import glob
import platform
import subprocess

from setuptools import Extension, setup

NAME = 'n2'
VERSION = '0.1.7'


def long_description():
    with io.open('README.rst', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    image_directive = 'image:: '
    for i in range(len(lines)):
        directive_start = lines[i].find(image_directive)
        is_absolute_url = any(x in lines[i] for x in ['https://', 'http://'])
        if directive_start != -1 and not is_absolute_url:
            directive_end = directive_start + len(image_directive)
            lines[i] = lines[i][:directive_end] + 'https://raw.githubusercontent.com/kakao/n2/master/' + lines[i][directive_end:]
    readme = ''.join(lines)

    return readme


def set_binary_mac():
    gcc_dir = subprocess.check_output('brew --prefix gcc', shell=True).decode().strip()
    gcc_dir = os.path.join(gcc_dir, 'bin')
    gpp_binaries = glob.glob(os.path.join(gcc_dir, 'g++-[0-9]*'))
    gcc_binaries = glob.glob(os.path.join(gcc_dir, 'gcc-[0-9]*'))
    binaries = [gcc_binaries, gpp_binaries]
    targets = ['CC', 'CXX']
    for binary, target in zip(binaries, targets):
        if binary:
            binary = sorted(binary, key=lambda x: int(x.split('-')[1]))[-1]
            os.environ[target] = os.path.join(gcc_dir, binary)
        else:
            msg = ('\n  \033[1;31mNo gcc available.\033[37m Install gcc from'
                   ' \033[32mHomebrew\033[37m using `\033[32mbrew install gcc\033[37m`.\033[0m\n')
            sys.exit(msg)


def is_buildable():
    try:
        for option, flag in zip(['C++14', 'OpenMP'], ['-std=c++14', '-fopenmp']):
            for cmd, env in zip(['gcc', 'g++'], ['CC', 'CXX']):
                cmd = os.environ.get(env) or cmd
                test_cmd = 'echo "int main(){}" | ' + cmd + ' -fsyntax-only ' + flag + ' -xc++ -'
                subprocess.check_output(test_cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError:
        msg = ('\n  \033[1;37mYour compiler(\033[33m\"%s\"\033[37m) may not support \033[31m\"%s\"\033[0m.'
               '\n  \033[1mSet CC, CXX environment variable as suitable gcc.\033[0m\n') % (cmd, option)
        return False, msg
    return True, None


def define_extensions(**kwargs):
    system = platform.system().lower()
    if 'windows' in system:  # Windows
        sys.exit('Installation on Windows is not supported yet.')
    elif 'darwin' in system:  # osx
        is_buildable()[0] or set_binary_mac()

    able, fail_msg = is_buildable()
    if not able:
        sys.exit(fail_msg)

    libraries = []
    extra_link_args = []
    extra_compile_args = ['-std=c++14', '-O3', '-fPIC', '-march=native', '-DNDEBUG', '-DBOOST_DISABLE_ASSERTS']
    extra_link_args.append('-fopenmp')
    extra_compile_args.append('-fopenmp')

    sources = ['./src/heuristic.cc', './src/hnsw.cc', './src/hnsw_node.cc',
               './src/hnsw_build.cc', './src/hnsw_model.cc', './src/hnsw_search.cc',
               './src/mmap.cc', './bindings/python/n2.pyx']

    boost_dirs = ['assert', 'bind', 'concept_check', 'config', 'core', 'detail', 'heap', 'iterator', 'mp11', 'mpl',
                  'parameter', 'preprocessor', 'static_assert', 'throw_exception', 'type_traits', 'utility']
    include_dirs = ['./include/', './third_party/spdlog/include/', './third_party/eigen']
    include_dirs.extend(['third_party/boost/' + b + '/include/' for b in boost_dirs])

    return Extension(name='n2',
                     sources=sources,
                     extra_compile_args=extra_compile_args,
                     libraries=libraries,
                     extra_link_args=extra_link_args,
                     include_dirs=include_dirs,
                     language='c++',)


setup(
    name=NAME,
    version=VERSION,
    description='Approximate Nearest Neighbor library',
    long_description=long_description(),
    author='Kakao.corp',
    author_email='recotech.kakao@gmail.com',
    license='Apache License 2.0',
    setup_requires=[
        'setuptools>=18',
        'cython',
    ],
    install_requires=[
        'cython'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Topic :: Software Development :: Libraries :: Python Modules'],

    keywords='Approximate Nearest Neighbor',
    ext_modules=[
        define_extensions(),
    ]
)
