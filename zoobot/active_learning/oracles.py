"""
Oracle for active learning.
Available as either:
- MockPanoptes, which uses a simulated catalog to return new (requested) labels
- Panoptes, which uses a running stream from Panoptes to request and return real classifications
"""
import os
import json
import logging

import pandas as pd
from shared_astro_utils import upload_utils
from gzreduction.main import Volunteers

from zoobot.science_logic import define_experiment

class Oracle():

    def request_labels(self, top_acquisition_ids, name):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError


class Panoptes(Oracle):

    def __init__(self, catalog_loc, login_loc, project_id, workflow_ids, last_id, question):
        assert os.path.exists(catalog_loc)  # in principle, should only need unlabelled galaxies
        self._catalog_loc = catalog_loc  # unlabelled catalog
        self._login_loc = login_loc
        self._project_id = project_id
        self._workflow_ids = workflow_ids
        self._full_catalog = pd.read_csv(catalog_loc)  # e.g. joint catalog with file locs
        # all catalog columns will be uploaded, be careful
        self.last_id = last_id  # ignore classifications before this id TODO REMOVE
        # '91178981' for first DECALS classification, bad idea - >1 million responses!
        # 'TODO' for last id as of last major reduction, 2nd April 2019
        self.question = question  # e.g. 'smooth', 'bar'

        if os.path.isdir('/home/ubuntu/root'):
            logging.info('Running on AWS')
            working_dir = '/home/ubuntu/root/repos/zoobot/data/decals/classifications/streaming'
        else:
            working_dir = '/data/repos/zoobot/data/decals/classifications/streaming'
        assert os.path.isdir(working_dir)
        self._volunteers = Volunteers(
            working_dir=working_dir,
            workflow_ids=self._workflow_ids,
            max_classifications=1e8
        )

    def request_labels(self, subject_ids, name, retirement):
        """Upload subjects with ids matching subject_ids to Panoptes project
        
        Args:
            subject_ids ([type]): [description]
        """
        selected_catalog = self._full_catalog[self._full_catalog['id_str'].isin(subject_ids)]  # getting really messy with this...
        upload_utils.upload_to_gz(
            login_loc=self._login_loc,
            selected_catalog=selected_catalog,
            name=name,
            retirement=retirement,
            project_id=self._project_id,
            uploader='panoptes_oracle'
        )

    def get_labels(self, working_dir):
        """Get all recent labels from Panoptes. 
        - (Download with Panoptes Python client
        - Aggregate with GZ Reduction Spark routine)
        """
        raise NotImplementedError('Needs updating for multi-question')
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

        all_classifications = self._volunteers.get_all_classifications()
        # this now gets ALL (retired) labels, not just new ones - be careful when using!

        # science logic needs to remain inside define_experiment, not here
        retired, _ = define_experiment.split_retired_and_not(all_classifications, self.question)
        retired = define_experiment.define_identifiers(retired)  # add iauname
        retired = define_experiment.define_labels(retired, self.question)  # add 'label' and 'total_votes', drop low n bars
        retired = define_experiment.drop_duplicates(retired)

        logging.info('Labels acquired from oracle (including old): {}'.format(len(retired)))
        return retired['id_str'].values, retired['label'].values, retired['total_votes'].values


    def save(self, save_dir):
        data = {
            'catalog_loc': self._catalog_loc,
            'login_loc': self._login_loc,
            'project_id': self._project_id,
            'workflow_ids': self._workflow_ids,
            'last_id': self.last_id,
            'question': self.question
        }
        with open(os.path.join(save_dir, 'oracle_config.json'), 'w') as f:
            json.dump(data, f)

class PanoptesMock(Oracle):

    def __init__(self, oracle_loc, subjects_requested_loc, label_cols):
        assert os.path.isfile(oracle_loc)  # must already exist
        logging.info('Using oracle loc: {}'.format(oracle_loc))
        logging.info('Using subjects requested loc: {}'.format(subjects_requested_loc))
        self._oracle_loc = oracle_loc
        self._subjects_requested_loc = subjects_requested_loc
        self._label_cols = label_cols

    def request_labels(self, subject_ids, name, retirement):
        logging.info('Pretending to upload {} subjects: {}'.format(len(subject_ids), name))
        assert len(set(subject_ids)) == len(subject_ids)  # must be unique
        with open(self._subjects_requested_loc, 'w') as f:
            json.dump(subject_ids, f)

    def get_labels(self, working_dir):
        # oracle.csv is created by make_shards.py, contains label and id_str pairs of vote fractions
        if not os.path.isfile(self._subjects_requested_loc):
            logging.warning(
                'No previous subjects requested at {}'.format(self._subjects_requested_loc))
            return [], []  # must unpack 3 values, look here if 'not enough values to unpack' error

        with open(self._subjects_requested_loc, 'r') as f:
            subject_ids = json.load(f)
        assert isinstance(subject_ids, list)
        assert len(subject_ids) > 0
        assert len(set(subject_ids)) == len(subject_ids)  # must be unique
        os.remove(self._subjects_requested_loc)

        # using the mock oracle catalog saved in define_experiment.py, that includes labels for 'unlabelled' subjects
        known_catalog = pd.read_csv(
            self._oracle_loc,
            usecols=['id_str'] + self._label_cols,
            dtype={'id_str': str}
        )
        # return labels from the oracle, mimicking live GZ classifications
        id_str_dummy_df = pd.DataFrame(data={'id_str': subject_ids})
        logging.info(f'{len(id_str_dummy_df)} id strs in dummy df')
        logging.info('e.g. {}'.format(id_str_dummy_df['id_str'][:5].values))
        logging.info(f'{len(known_catalog)} in known catalog at {self._oracle_loc}')
        logging.info('with ids e.g. {}'.format(known_catalog['id_str'][:5].values))
        matching_df = pd.merge(id_str_dummy_df, known_catalog, how='inner', on='id_str')
        logging.info(f'{len(matching_df)} matches in known catalog')
        if not len(id_str_dummy_df) == len(matching_df):
            missing_ids = set(id_str_dummy_df['id_str']) - set(matching_df['id_str'])
            # if this fails, some to-be-queried id_strs are not in the oracle catalog
            print(f'Missing ids: {missing_ids}')
            raise ValueError(f'{len(missing_ids)} ids not found in oracle catalog')
        labels = list(matching_df[self._label_cols].to_dict(orient='records'))
        # ensure these are explicitly floats, or tf will complain when loading them
        for label_dict in labels:
            for k in label_dict.keys():
                label_dict[k] = float(label_dict[k])
        logging.info(f'{len(labels)} matching labels returned from known catalog')
        assert len(subject_ids) == len(labels)
        return subject_ids, labels

    def save(self, save_dir):
        data = {
            'oracle_loc': self._oracle_loc,
            'subjects_requested_loc': self._subjects_requested_loc,
            'label_cols': self._label_cols
        }
        with open(os.path.join(save_dir, 'oracle_config.json'), 'w') as f:
            json.dump(data, f)


def load_panoptes_oracle(save_dir):
    with open(os.path.join(save_dir, 'oracle_config.json'), 'r') as f:
        return Panoptes(**json.load(f))

def load_panoptes_mock_oracle(save_dir):
    with open(os.path.join(save_dir, 'oracle_config.json'), 'r') as f:
        return PanoptesMock(**json.load(f))

def load_oracle(save_dir):
    try:
        return load_panoptes_oracle(save_dir)
    except TypeError:  # will have different args for init
        return load_panoptes_mock_oracle(save_dir)

