import os

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR




def get_labels(subject_ids):
    # created by run_active_learning.py
    catalog_loc = os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv')
    known_catalog = pd.read_csv(catalog_loc, usecols=['id_str', 'label'], dtype={'id_str': str}) 
    # mimic GZ
    labels = []
    for id_str in subject_ids:
        matching_row = known_catalog[known_catalog['id_str'] == id_str].iloc[0]
        labels.append(matching_row['label'])
    assert len(subject_ids) == len(labels)
    return labels