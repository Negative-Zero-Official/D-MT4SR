# -*- coding: utf-8 -*-
"""Verify that this folder is a complete, self-contained handoff package.

Run this on the SENDER's machine before copying, and again on the RECEIVER's
machine after copying. It checks that every module the code imports is
present, that the preprocessed dataset is here and loads, that the
documentation and preflight script are here, and that nothing the code needs
lives outside this folder.

    python check_package.py                    # check the current folder
    python check_package.py --path handoff/    # check a folder being assembled
    python check_package.py --skip-load        # structure only, don't load the .npy

Exits non-zero if anything required is missing.
"""
import argparse
import ast
import os
import sys

# Files that must exist for the suite to run at all.
REQUIRED_CODE = [
    'main.py', 'trainers.py', 'modules.py', 'seqmodels.py', 'datasets.py',
    'utils.py', 'run_experiments.py', 'aggregate_results.py', 'preflight.py',
]
REQUIRED_DOCS = ['README_HANDOFF.md', 'RUN_COMMANDS.txt', 'requirements.txt']
REQUIRED_DATA = ['data/Office_ProductsPartitioned_5core.npy']
# Present is good, absent is fine.
OPTIONAL = ['verify_chunking.py', 'data/preprocess_fromscratch.py', 'README.md']

# Third-party imports the code is allowed to make. Anything imported that is
# neither stdlib, third-party-allowed, nor a local .py file in the package is
# reported as an unresolved dependency.
THIRD_PARTY = {'numpy', 'scipy', 'torch', 'tqdm', 'matplotlib', 'pandas'}

STDLIB = set(getattr(sys, 'stdlib_module_names', ())) | {
    'argparse', 'ast', 'os', 'sys', 'time', 'json', 'math', 'random', 'pickle',
    'platform', 'subprocess', 'traceback', 'collections', 'types', 'copy',
    'itertools', 'functools', 'shutil', 'glob', 're', 'datetime', 'gzip',
    'multiprocessing', 'warnings', 'statistics', 'csv', 'pathlib',
}

problems = []
warnings = []


def check(label, ok, detail='', fatal=True):
    mark = 'ok  ' if ok else ('MISS' if fatal else 'warn')
    print(f'  [{mark}] {label}{("  " + detail) if detail else ""}')
    if not ok:
        (problems if fatal else warnings).append(label)
    return ok


def top_level_imports(path):
    """Module names imported by a python file, at any nesting level."""
    try:
        tree = ast.parse(open(path, encoding='utf-8').read())
    except (SyntaxError, UnicodeDecodeError) as exc:
        problems.append(f'{path} does not parse: {exc}')
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # relative import; resolves inside the package
                continue
            if node.module:
                names.add(node.module.split('.')[0])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='.', help='folder to check (default: .)')
    ap.add_argument('--skip-load', action='store_true',
                    help='do not actually load the .npy (faster, less thorough)')
    cli = ap.parse_args()

    root = os.path.abspath(cli.path)
    if not os.path.isdir(root):
        print(f'No such folder: {root}')
        return 1
    os.chdir(root)

    print()
    print('=' * 74)
    print('PACKAGE CHECK')
    print(f'{root}')
    print('=' * 74)

    print('\nCode:')
    for f in REQUIRED_CODE:
        check(f, os.path.exists(f))

    print('\nDocumentation:')
    for f in REQUIRED_DOCS:
        check(f, os.path.exists(f))

    print('\nData:')
    for f in REQUIRED_DATA:
        exists = os.path.exists(f)
        size = f'({os.path.getsize(f) / 1e6:.0f} MB)' if exists else ''
        check(f, exists, size)

    print('\nOptional:')
    for f in OPTIONAL:
        check(f, os.path.exists(f), fatal=False)

    # ---------------------------------------------------------------
    # Every import must resolve to stdlib, an allowed third-party
    # package, or a .py file sitting in this folder.
    # ---------------------------------------------------------------
    print('\nImport closure (nothing may resolve outside this folder):')
    local_modules = {f[:-3] for f in os.listdir('.') if f.endswith('.py')}
    unresolved = {}
    for f in REQUIRED_CODE + [o for o in OPTIONAL if o.endswith('.py')]:
        if not os.path.exists(f):
            continue
        for name in top_level_imports(f):
            if name in STDLIB or name in THIRD_PARTY or name in local_modules:
                continue
            unresolved.setdefault(name, []).append(f)
    if unresolved:
        for name, where in sorted(unresolved.items()):
            check(f'unresolved import {name!r} (from {", ".join(where)})', False)
    else:
        check(f'all imports resolve ({len(local_modules)} local modules, '
              f'{len(THIRD_PARTY)} third-party allowed)', True)

    # ---------------------------------------------------------------
    # Hardcoded paths that would point outside the package.
    # ---------------------------------------------------------------
    print('\nHardcoded paths:')
    data_path_ok = False
    if os.path.exists('utils.py'):
        src = open('utils.py', encoding='utf-8').read()
        data_path_ok = "np.load('./data/'" in src
    check("utils.get_user_seqs_MoHRdata() loads from ./data/", data_path_ok,
          'so the suite MUST be run from this folder')

    suspicious = []
    for f in REQUIRED_CODE:
        if not os.path.exists(f):
            continue
        for i, line in enumerate(open(f, encoding='utf-8'), 1):
            s = line.strip()
            if s.startswith('#'):
                continue
            if 'C:\\Users' in s or '/home/' in s or os.path.expanduser('~') in s:
                suspicious.append(f'{f}:{i}')
    check('no absolute machine-specific paths in the code',
          not suspicious, ', '.join(suspicious[:4]) if suspicious else '')

    # ---------------------------------------------------------------
    # The dataset must actually load and report the expected shape.
    # ---------------------------------------------------------------
    if not cli.skip_load and os.path.exists(REQUIRED_DATA[0]):
        print('\nDataset load:')
        try:
            sys.path.insert(0, root)
            from utils import get_user_seqs_MoHRdata
            (user_seq, max_item, _, _, num_users, _, rel_map, Item,
             times) = get_user_seqs_MoHRdata('Office_Products')
            check('Office_Products loads', True)
            check(f'users = {num_users:,}', num_users > 0)
            check(f'items = {max_item + 2:,}', max_item > 0)
            check(f'relationship types = {len(rel_map)} '
                  f'({", ".join(map(str, rel_map))})', len(rel_map) > 0)
            check(f'real timestamps present = {times is not None}',
                  True, fatal=False)
        except Exception as exc:
            check(f'Office_Products loads ({type(exc).__name__}: {exc})', False)
    elif cli.skip_load:
        print('\nDataset load:  skipped (--skip-load)')

    # ---------------------------------------------------------------
    print()
    print('=' * 74)
    if problems:
        print(f'INCOMPLETE -- {len(problems)} problem(s):')
        for p in problems:
            print(f'  - {p}')
        print()
        print('This package is NOT ready to send. Add the missing pieces and re-run.')
        print('=' * 74)
        return 1
    print('COMPLETE -- the package is self-contained.')
    if warnings:
        print(f'({len(warnings)} optional item(s) absent: {", ".join(warnings)})')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
