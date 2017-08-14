#!/usr/bin/env python3

import os
import sys

def pick_or_default(msg, default):
    value = input(f'{msg} [{default}]: ')
    return value if value else default

def create_export_str(varname, addend, keep_old=True):
    prefix = f'export {varname.upper()}="{addend}'
    infix = f':${varname.upper()}' if keep_old else ''
    suffix = '"'
    return prefix + infix + suffix

def split_filter_env(key, default_path='~', cond=''):
    if key in os.environ:
        paths = [i for i in os.environ[key].split(':') if cond in i]
        if paths:
            default_path = paths[0]

    return os.path.expanduser(default_path)

def determine(varname, msg, default=('~',), filter=('',), keep=True):
    path = split_filter_env(varname.upper(),
                            os.path.join(*default),
                            os.path.join(*filter))
    path = pick_or_default(f'{msg} ({varname.upper()})', path)
    return create_export_str(varname, path, keep)

def main():
    try:
        path = determine('path', 'Conda bin path',
                         ('~', 'anaconda3', 'bin'),
                         ('conda', 'bin'))
        cpath = determine('cpath', 'Cuda include path',
                         ('~', 'cuda', 'include'),
                         ('cuda', 'include'))
        lpath = determine('library_path', 'Cuda lib path',
                         ('~', 'cuda', 'lib64'),
                         ('cuda', 'lib'))
        ldpath = determine('ld_library_path', 'Cuda lib loaded path',
                         ('~', 'cuda', 'lib64'),
                         ('cuda', 'lib'))
        datadir = determine('data_dir', 'Data directory',
                            ('~', 'data'), keep=False)
    except KeyboardInterrupt:
        print('Aborted. No output file was written.')
    with open(os.path.join('.', 'tools', 'grid', 'startup.sh'), 'w') as f:
        for p in [path, cpath, lpath, ldpath, datadir]:
            print(p, file=f)

if __name__ == '__main__':
    main()
