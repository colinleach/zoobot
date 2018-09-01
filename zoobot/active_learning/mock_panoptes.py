import os

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR


def get_labels(subject_ids):
    # created by run_active_learning.py
    catalog_loc = os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv')
    label_split_value = 0.4
    known_catalog = pd.read_csv(catalog_loc, usecols=['id_str', 'smooth-or-featured_smooth_fraction'], dtype={'id_str': str})
    known_catalog['label'] = (known_catalog['smooth-or-featured_smooth_fraction'] > label_split_value).astype(int)
    # mimic GZ
    labels = []
    for id_str in subject_ids:
        matching_rows = known_catalog[known_catalog['id_str'] == id_str]
        assert len(matching_rows) > 0
        matching_row = matching_rows.iloc[0]
        labels.append(matching_row['label'])
    assert len(subject_ids) == len(labels)
    return labels