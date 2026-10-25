"""Retrieve the public dataset or verify an existing local snapshot by SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, entry):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise ValueError('Source snapshot mismatch: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(os.environ.get('IOWA_DRS_DATA_DIR', ROOT / 'data/raw')))
    parser.add_argument('--check-existing', type=Path,
                        help='Read-only check of an existing snapshot; no download or copy.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/public_source_manifest.json').read_text(encoding='utf8'))
    folder = (args.check_existing or args.output_dir).expanduser().resolve()
    if args.check_existing is None:
        folder.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        path = folder / entry['name']
        if path.exists() or args.check_existing is not None:
            verify(path, entry)
        else:
            temporary = path.with_name(path.name + '.part')
            request = urllib.request.Request(entry['source_url'], headers={'User-Agent': 'Iowa-DRS-reanalysis/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as target:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            verify(temporary, entry)
            temporary.replace(path)
        print('verified', entry['name'])
    print('Verified', len(manifest['files']), 'public attachments for DOI', manifest['dataset_doi'])


if __name__ == '__main__':
    main()
