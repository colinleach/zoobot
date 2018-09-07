import os
import json

import pytest

from zoobot.tests import TEST_FIGURE_DIR
from zoobot.active_learning import analysis


@pytest.fixture()
def tfrecord_index_loc(tfrecord_dir, example_tfrecord_loc):
    loc = os.path.join(tfrecord_dir, 'tfrecord_index.json')
    with open(loc, 'w') as f:
        json.dump([example_tfrecord_loc, example_tfrecord_loc], f)
    return loc

def test_show_subjects_by_iteration(tfrecord_index_loc, size, channels):
    save_loc = os.path.join(TEST_FIGURE_DIR, 'subjects_in_shards.png')
    analysis.show_subjects_by_iteration(tfrecord_index_loc, 5, size, channels, save_loc)
