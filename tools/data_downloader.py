"""
Reads the docs/datasets.md file and lists the contents as well as allows
downloading them.

Usage:

    python data_downloader.py [-h] [-l] ...

    -l   lists all available datasets and their keys
    -h   prints the help text
    ...  can be one or many identifiers of datasets to be downloaded. If
         "all", all datasets will be downloaded.
         Existing files are skipped!
"""
import argparse
import os
import re
import sys
import textwrap
import tarfile
import shutil

from contextlib import closing
from multiprocessing import Pool

import requests


class Dataset:
    """Wraps a dataset."""

    def __init__(self, entry):
        """Initializes the dataset."""
        self.__parse_entry(entry)

    def __parse_entry(self, entry):
        """Parses the markdown entry.

        entry should be a list of markdown lines.

        The markdown file should contain entries like this:
            (no dashed lines)

        --------------------------------------------------
        # Name (key)

        > description
        > longer description

        - Size: <filesize>
        - URL: <url>

        ```tex
        <bibtex references>
        ```

        # Files

        - <file url>
        - <file url>
        - ...
        --------------------------------------------------
        """
        matches = re.search('\# (.+) [(](.*)[)].*', entry[0])
        self.title, self.key = matches.group(1), matches.group(2)
        self.description = ''
        self.refs = []
        self.file_urls = []

        in_refs = False
        in_files = False
        for line in entry[1:]:
            if not line:  # skip empty lines
                pass
            elif line.startswith('> '):  # descriptions are quotes
                self.description += line[2:] + ' '
            elif in_refs and line.startswith('@'):  # new bib entry
                try:
                    self.refs[-1] += '\n'
                except IndexError:
                    pass
                self.refs.append(line)
            elif line.startswith('```'):  # toggle bib entries
                in_refs = not in_refs
            elif in_refs:  # append to bib entry
                self.refs[-1] += '\n' + line
            elif line.startswith('## Files'):  # finally: come to files
                in_files = True
            elif in_files:  # in file list
                self.file_urls.append(line[2:])
            elif line.startswith('- Size: '):  # size
                start = len('- Size: ')
                self.size = line[start:]
            elif line.startswith('- URL: '):  # url
                start = len('- URL: ')
                self.url = line[start:]

    def __format__(self, format_spec='8'):
        """Returns a description of this dataset."""
        indent = ' ' * int(format_spec)
        start = f'{self.key:{format_spec}}{self.title}, ~{self.size}',
        wrapper = textwrap.TextWrapper(initial_indent=indent,
                                       subsequent_indent=indent,
                                       replace_whitespace=False)
        content = f'{self.description}\n{self.url}'
        return '\n'.join([*start] + wrapper.wrap(content))


def read_datasets():
    """Reads the datasets from docs/datasets.md

    Returns a list of datasets.
    """
    with open('docs/datasets.md', 'r') as ds_file:
        entries = []
        for line in ds_file.read().splitlines():
            if line.startswith('# '):
                entry = [line]
                entries.append(entry)
            else:
                entry.append(line)
    return sorted((Dataset(e) for e in entries), key=lambda ds: ds.key)


def download_file(url_path):
    """Downloads a file from a url to the specified path."""
    url, path = url_path

    try:
        local_size = round(os.stat(path).st_size / 2 ** 20, 2)
        print(f'LocalSize is {local_size}')
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
        local_size = -1

    with closing(requests.get(url, stream=True, timeout=20)) as request:
        try:
            remote_size = round(
                int(request.headers['Content-Length']) / 2 ** 20, 2)
            unit = 'MB'
        except KeyError as e:
            print(request.headers)
            return f'Unable to download {url}, KeyError: {e}'
        if local_size == remote_size:
            return f'Skipped {url}: Size of {remote_size} {unit} would be unchanged.'
        with open(path, 'wb') as local_file:
            print(
                f'Starting download of {remote_size} {unit} from {url} to {path}.')
            for content in request.iter_content(chunk_size=1024):
                local_file.write(content)
    return f'Downloaded {remote_size} {unit} from {url} to {path}.'


def download_dataset(dataset):
    fn = lambda url: os.path.join(
        'data', dataset.key, 'raw', os.path.basename(url))
    with Pool(4) as pool:
        results = pool.map(download_file,
                           [(url, fn(url)) for url in dataset.file_urls])
    print('Results:', end='\n\t')
    print(*results, sep='\n\t')

    if results:
        unpack_dataset(os.path.join('data', dataset.key, 'raw'))


def download_datasets(datasets):
    data = [d for d in read_datasets() if 'all' in datasets or d.key in datasets]
    if not data:
        print('Nothing to download.')
        return
    print('Preparing to download the following dataset(s):', end='\n\t')
    print(*(d.key for d in data), sep='\n\t', end='\n\n')

    for d in data:
        download_dataset(d)


def unpack_dataset(path):
    """
    Unpacks all compressed data files.
    Args:
        path: the folder that contains all packaged files
    """
    destfolder = os.path.join(os.path.dirname(path), 'unpacked')
    print(f'Extract archives to {destfolder}')
    for item in os.listdir(path):
        if item.endswith('tar.gz') or item.endswith('tgz'):
            print(f'Extracting {item}')
            with tarfile.open(os.path.join(path, item)) as tar:
                tar.extractall(os.path.join(destfolder, item[:item.find('.')]))
        else:
            if not os.path.isdir(destfolder):
                os.makedirs(destfolder)
            shutil.copy2(os.path.join(path, item),
                         os.path.join(destfolder, item))
    flatten_folders(destfolder)


def flatten_folders(path):
    """
    Normalizes a folder such that it's subfolders contain no more folders but only files.
    Args:
        path: the top folder in which the subfolders should be normalized.
    """
    if os.path.isdir(path):
        for item in os.listdir(path):
            for root, dirs, files in os.walk(os.path.join(path, item)):
                if len(dirs) > 0:
                    for subdir in dirs:
                        files = os.listdir(os.path.join(path, item, subdir))
                        for f in files:
                            dest = os.path.join(root)
                            source = os.path.join(root, subdir, f)
                            if os.path.isdir(source):
                                files2 = os.listdir(
                                    os.path.join(path, item, subdir, f))
                                for f2 in files2:
                                    dest = os.path.join(root)
                                    source = os.path.join(root, subdir, f, f2)
                                    shutil.copy2(source, dest)
                            else:
                                shutil.copy2(source, dest)
                        shutil.rmtree(os.path.join(path, item, subdir))


def list_datasets():
    """Prints a list of all available datasets."""
    datasets = read_datasets()
    keylength = max([len(ds.key) for ds in datasets]) + 3
    print('\n\n'.join(format(ds, str(keylength)) for ds in datasets))


def main(list_only=False, datasets=None):
    """Either lists the files (list_only == True) or downloads the
    specified datasets."""
    list_datasets() if list_only else download_datasets(datasets or [])


def parse_args():
    """Parses the command line arguments.

    Returns:
        The resulting Namespace object.
    """
    parser = argparse.ArgumentParser(description='Data downloader for training \
                                     and test data')
    parser.add_argument('-l', '--list', action='store_const', const=True,
                        help='Only list all available datasets.')
    parser.add_argument('data', nargs=argparse.REMAINDER, help='Dataset \
                        identifiers to identify data to be downloaded to \
                        data/. For a list of available datasets, see -l.')
    args = parser.parse_args()
    if not args.data and not args.list:
        sys.exit(parser.print_help())

    return args.list, args.data


if __name__ == '__main__':
    main(*parse_args())
