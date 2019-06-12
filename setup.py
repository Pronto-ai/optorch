import os
import re
import sys
import sysconfig
import platform
import subprocess
import shutil
import inspect

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

def get_torch_path():
    import torch
    return os.path.dirname(inspect.getfile(torch))

class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        return 'cp37', 'cp37m', 'manylinux1_x86_64'
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not os.path.exists(extdir):
            os.makedirs(extdir)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cmake_args += ['-DTORCH_PATH=' + get_torch_path()]
        print('cmake args:', cmake_args)

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        shutil.copy2(os.path.join(self.build_temp, 'main'), extdir)

setup(
    name='optorch',
    version='0.0.3',
    description='Nonlinear solver for PyTorch',
    long_description='See: https://github.com/pronto-ai/optorch',
    long_description_content_type='text/plain',
    url='https://github.com/pronto-ai/optorch',
    author='Pronto AI',
    license='MIT',
    packages=['optorch'],
    ext_modules=[CMakeExtension('optorch/cpp')],
    install_requires=['torch'],
    cmdclass={
        'build_ext':CMakeBuild,
        'bdist_wheel': bdist_wheel,
    },
)
