import copy

import pytest
import numpy as np

from zoobot.active_learning import make_shards, execute


# due to conftest.py, catalog fits_loc is absolute and points to fits_native_dir
@pytest.fixture()
def labelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_labelled'  # must be unique
    catalog['label'] = np.random.randint(0, 2, len(catalog))
    return catalog


@pytest.fixture()
def unlabelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_unlabelled'  # must be unique
    return catalog


@pytest.fixture()
def shard_config(tmpdir, size, channels):
    config = make_shards.ShardConfig(
        shard_dir=tmpdir.mkdir('base_dir').strpath,
        inital_size=size,
        final_size=size,
        channels=channels)
    return config


@pytest.fixture()
def shard_config_ready(shard_config, labelled_catalog, unlabelled_catalog):
    config = copy.copy(shard_config)
    config.prepare_shards(labelled_catalog, unlabelled_catalog)
    assert config.ready()
    return config


@pytest.fixture()
def active_config(shard_config_ready, tmpdir):
    config = execute.ActiveConfig(
        shard_config_ready, 
        run_dir=tmpdir.mkdir('run_dir').strpath,
        iterations=2,
        shards_per_iter=2,
        subjects_per_iter=10,
        warm_start=True
        )

    return config


@pytest.fixture()
def active_config_ready(active_config):
    config = copy.copy(active_config)
    config.prepare_run_folders()
    assert config.ready()
    return config
