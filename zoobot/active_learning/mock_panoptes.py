import os
import json
import logging

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR

DIR_OF_THIS_FILE = os.path.dirname(os.path.realpath(__file__))
SUBJECTS_REQUESTED = os.path.join(DIR_OF_THIS_FILE, 'subjects_requested.json')
# delete before each script execution, don't cross-contaminate
if os.path.isfile(SUBJECTS_REQUESTED):
    os.remove(SUBJECTS_REQUESTED)


def request_labels(subject_ids):
    with open(SUBJECTS_REQUESTED, 'w') as f:
        json.dump(subject_ids, f)


def get_labels():
    # oracle.csv is created by make_shards.py, contains label and id_str pairs of vote fractions
    if not os.path.isfile(SUBJECTS_REQUESTED):
        logging.warning('No previous subjects requested at {}'.format(SUBJECTS_REQUESTED))
        return [], []

    with open(SUBJECTS_REQUESTED, 'r') as f:
        subject_ids = json.load(f)
    os.remove(SUBJECTS_REQUESTED)

    oracle_loc = os.path.join(DIR_OF_THIS_FILE, 'oracle_gz2.csv')
    known_catalog = pd.read_csv(oracle_loc, usecols=['id_str', 'label'], dtype={'id_str': str, 'label': float})
    # return labels from the oracle, mimicking live GZ classifications
    labels = []
    for id_str in subject_ids:
        matching_rows = known_catalog[known_catalog['id_str'] == id_str]
        assert len(matching_rows) > 0  # throw error if id_str not recognised by oracle
        matching_row = matching_rows.iloc[0]
        labels.append(matching_row['label'])
    assert len(subject_ids) == len(labels)
    return subject_ids, labels
