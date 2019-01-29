import os
import json
import logging

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR

SHARD_DIR = 'data/gz2_shards/uint8_256px_bar_n'
# DIR_OF_THIS_FILE = os.path.dirname(os.path.realpath(__file__))
# ORACLE_LOC = os.path.join(DIR_OF_THIS_FILE, 'oracle_gz2_bar.csv')
ORACLE_LOC = os.path.join(SHARD_DIR, 'oracle_gz2_bar.csv')
# ORACLE_LOC = 'this_should_fail'
# assert os.path.isfile(ORACLE_LOC)

# SUBJECTS_REQUESTED = os.path.join(DIR_OF_THIS_FILE, 'subjects_requested.json')
# # delete before each script execution, don't cross-contaminate
# if os.path.isfile(SUBJECTS_REQUESTED):
#     os.remove(SUBJECTS_REQUESTED)


SUBJECTS_REQUESTED = 'data/gz2_shards/runs_cache/many_random_subjects.json'

def request_labels(subject_ids):
    with open(SUBJECTS_REQUESTED, 'w') as f:
        json.dump(subject_ids, f)


def get_labels():
    # oracle.csv is created by make_shards.py, contains label and id_str pairs of vote fractions
    if not os.path.isfile(SUBJECTS_REQUESTED):
        logging.warning(
            'No previous subjects requested at {}'.format(SUBJECTS_REQUESTED))
        return [], []

    with open(SUBJECTS_REQUESTED, 'r') as f:
        subject_ids = json.load(f)
    assert isinstance(subject_ids, list)
    assert len(subject_ids) > 0
    os.remove(SUBJECTS_REQUESTED)

    known_catalog = pd.read_csv(
        ORACLE_LOC,
        usecols=['id_str', 'label'],
        dtype={'id_str': str, 'label': float}
    )
    # return labels from the oracle, mimicking live GZ classifications
    labels = []
    # TODO could make this search far quicker by joining, not searching many times
    for id_str in subject_ids:
        matching_rows = known_catalog[known_catalog['id_str'] == id_str]
        # throw error if id_str not recognised by oracle
        assert len(matching_rows) > 0
        matching_row = matching_rows.iloc[0]
        labels.append(matching_row['label'])
    assert len(subject_ids) == len(labels)
    return subject_ids, labels


if __name__ == '__main__':
    # fill out subjects_requested so that we acquire many new random shards
    unlabelled_catalog = pd.read_csv(os.path.join(SHARD_DIR, 'unlabelled_catalog.csv'))
    subject_ids = list(unlabelled_catalog['id_str'].astype(str))  # entire unlabelled catalog!
    request_labels(subject_ids)  # will write to updated loc
