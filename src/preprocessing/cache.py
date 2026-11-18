import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import tempfile

import numpy as np

from src.data.projections import ProjectionImages, projection_image


def file_checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def implementation_fingerprint(projector_source):
    source_root = Path(__file__).resolve().parents[1]
    paths = [
        source_root / 'preprocessing' / name
        for name in ['mpca.py', 'fold.py', 'cache.py']
    ]

    return {
        'projection_source_sha256': hashlib.sha256(projector_source.encode()).hexdigest(),
        'rgb_export_source_sha256': hashlib.sha256(inspect.getsource(projection_image).encode()).hexdigest(),
        'source_sha256': {path.name: file_checksum(path) for path in paths},
        'packages': {
            name: importlib.metadata.version(name)
            for name in ['numpy', 'scipy', 'scikit-image', 'nibabel', 'Pillow']
        },
    }


def source_identities(indices, records, root):
    result = []

    for index in indices:
        record = records[index]
        checksum = record.get('sha256')

        # Main runners verify recorded hashes before preparing any splits.
        if not checksum:
            checksum = file_checksum(root / record['relative_path'])

        result.append({'sample_id': record['sample_id'], 'sha256': checksum})

    return result


def cache_key(signature):
    payload = json.dumps(signature, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_projection(directory, partition, values, output_format, empty_shape=None):
    if output_format == 'npy':
        np.save(directory / f'{partition}.npy', values, allow_pickle=False)
        return {'shape': list(values.shape), 'dtype': values.dtype.str}

    (directory / partition).mkdir()
    paths = []
    image_shape = None

    for index, projection in enumerate(values):
        image = projection_image(projection)
        shape = [image.height, image.width, 3]

        if image.mode != 'RGB' or (image_shape is not None and image_shape != shape):
            raise ValueError('PNG export requires aligned RGB projections')

        image_shape = shape
        relative = f'{partition}/{index:06d}.png'
        image.save(directory / relative, format='PNG')
        paths.append(relative)

    if image_shape is None:
        if empty_shape is None:
            raise ValueError('Cannot export an empty training partition')

        image_shape = empty_shape

    return {'shape': [len(paths), *image_shape], 'dtype': np.dtype(np.uint8).str, 'paths': paths}


def read_projection(directory, partition, signature, descriptor, mmap):
    if signature['output_format'] == 'png':
        expected = [f'{partition}/{index:06d}.png' for index in range(len(signature[partition]))]

        if descriptor['paths'] != expected or descriptor['dtype'] != np.dtype(np.uint8).str:
            raise ValueError('Cache PNG order or pixel dtype changed')

        return ProjectionImages([directory / path for path in expected], descriptor['shape'])

    return np.load(
        directory / f'{partition}.npy',
        allow_pickle=False,
        mmap_mode='r' if mmap else None,
    )


def read_entry(directory, signature, mmap):
    try:
        metadata = json.loads((directory / 'entry.json').read_text())

        if metadata['signature'] != signature or metadata['key'] != cache_key(signature):
            raise ValueError('Cache signature does not match the requested split')

        files = metadata['files']
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob('*') if path.is_file()
        }

        if actual_files != set(files) | {'entry.json'}:
            raise ValueError('Cache file inventory changed')

        for relative, checksum in files.items():
            path = (directory / relative).resolve()

            if not path.is_relative_to(directory.resolve()) or file_checksum(path) != checksum:
                raise ValueError(f'Cache checksum mismatch: {relative}')

        projections = []

        for partition in ['train', 'held_out']:
            descriptor = metadata['projections'][partition]
            projection = read_projection(directory, partition, signature, descriptor, mmap)

            if (
                list(projection.shape) != descriptor['shape']
                or len(projection.shape) != 4
                or projection.shape[-1] != 3
                or len(projection) != len(signature[partition])
            ):
                raise ValueError(f'Cache array layout mismatch: {partition}')

            if isinstance(projection, np.ndarray) and projection.dtype.str != descriptor['dtype']:
                raise ValueError(f'Cache array dtype mismatch: {partition}')

            projections.append(projection)

        fitted = json.loads((directory / 'pipeline' / 'metadata.json').read_text())

        if fitted['train_ids'] != [row['sample_id'] for row in signature['train']]:
            raise ValueError('Cached pipeline was not fitted on the requested training IDs')

        return projections

    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Corrupt projection cache: {directory}: {error}') from error


def cached_projections(root, signature, build, mmap=True, require_hit=False):
    if signature.get('output_format') not in {'png', 'npy'}:
        raise ValueError('Projection cache format must be png or npy')

    root = Path(root).expanduser().resolve()
    key = cache_key(signature)
    directory = root / key

    if directory.exists():
        projections = read_entry(directory, signature, mmap)
        return projections, directory, key, True

    if require_hit:
        raise FileNotFoundError(f'Prepared projection cache is missing: {directory}')

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{key}.', dir=root))

    try:
        partitions = build(staging / 'pipeline')
        descriptors = {}

        for partition, values in zip(['train', 'held_out'], partitions):
            empty_shape = descriptors['train']['shape'][1:] if 'train' in descriptors else None
            descriptors[partition] = write_projection(
                staging, partition, values, signature['output_format'], empty_shape=empty_shape,
            )

        del partitions

        metadata = {
            'key': key,
            'signature': signature,
            'projections': descriptors,
            'files': {
                path.relative_to(staging).as_posix(): file_checksum(path)
                for path in sorted(staging.rglob('*')) if path.is_file()
            },
        }
        (staging / 'entry.json').write_text(json.dumps(metadata, indent=2))
        read_entry(staging, signature, mmap=True)
        staging.rename(directory)

    except BaseException as error:
        # Keep incomplete builds separately for inspection; never publish them as hits.
        if staging.exists():
            (staging / 'failure.json').write_text(json.dumps({'error': repr(error)}, indent=2))
        raise

    return read_entry(directory, signature, mmap), directory, key, False
