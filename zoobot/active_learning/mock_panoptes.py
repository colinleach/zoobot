import os

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR


def get_labels(subject_ids):
    # oracle.csv is created by make_shards.py, contains label and id_str pairs of vote fractions
    dir_of_this_file = os.path.dirname(os.path.realpath(__file__))
    oracle_loc = os.path.join(dir_of_this_file, 'oracle.csv')
    known_catalog = pd.read_csv(oracle_loc, usecols=['id_str', 'label'], dtype={'id_str': str, 'label': float})
    # return labels from the oracle, mimicking live GZ classifications
    labels = []
    for id_str in subject_ids:
        matching_rows = known_catalog[known_catalog['id_str'] == id_str]
        assert len(matching_rows) > 0
        matching_row = matching_rows.iloc[0]
        labels.append(matching_row['label'])
    assert len(subject_ids) == len(labels)
    return labels
