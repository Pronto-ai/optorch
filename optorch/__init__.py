import os
import glob
import shutil
import subprocess

# skip patching when just building autodocs
if os.path.basename(os.getcwd()) != 'docsrc':
    patchelf = shutil.which('patchelf')
    if not patchelf:
        raise RuntimeError('You need patchelf to run optorch. (apt install patchelf)')

    import torch
    torch_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

    main_path = os.path.join(os.path.dirname(__file__), 'main')
    rpath = subprocess.check_output([patchelf, '--print-rpath', main_path])
    rpath = rpath.decode('utf-8').strip()

    if rpath != torch_path:
        print('patching rpath to', torch_path)
        subprocess.check_call([patchelf, '--set-rpath', torch_path, main_path])
        print('patched successfully')

from .problem import *
