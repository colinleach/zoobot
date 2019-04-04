import os
import shutil
import json
import logging
import time
from datetime import datetime

import pandas as pd
from shared_astro_utils import upload_utils, time_utils
import gzreduction

from zoobot.active_learning.oracle import Oracle
from zoobot.active_learning import prepare_catalogs, create_experiment_catalog

# Oracle state must not change between iterations! (Panoptes itself can change, of course)

class Panoptes(Oracle):

    def __init__(self, catalog_loc, login_loc, project_id, last_id, question):
        assert os.path.exists(catalog_loc)
        self._catalog_loc = catalog_loc
        self._login_loc = login_loc
        self._project_id = project_id
        self._full_catalog = pd.read_csv(catalog_loc)  # e.g. joint catalog with file locs
        # all catalog columns will be uploaded, be careful
        self.last_id = last_id  # ignore classifications before this id
        # '91178981' for first DECALS classification, bad idea - >2 million responses!
        # 'TODO' for last id as of last major reduction, 2nd April 2019
        self.question = question  # e.g. 'smooth', 'bar'

    def request_labels(self, subject_ids, name, retirement):
        """Upload subjects with ids matching subject_ids to Panoptes project
        
        Args:
            subject_ids ([type]): [description]
        """
        selected_catalog = self._full_catalog[self._full_catalog['id_str'].isin(subject_ids)]  # getting really messy with this...
        selected_catalog['retirement_limit'] = retirement
        logging.info('Uploading {} subjects to {}'.format(len(subject_ids), name))
        manifest = upload_utils.create_manifest_from_catalog(selected_catalog)
        upload_utils.upload_manifest_to_galaxy_zoo(
            subject_set_name=name,
            manifest=manifest,
            project_id=self._project_id,
            login_loc=self._login_loc)
        logging.info('Upload complete')

    def get_labels(self):
        """Get all recent labels from Panoptes. 
        - Download with Panoptes Python client
        - Aggregate with GZ Reduction Spark routine
        """
        responses_dir = '/tmp/recent_classifications'
        if os.path.isdir(responses_dir):
            shutil.rmtree(responses_dir)
        else:
            os.mkdir(responses_dir)
        # get raw API responses, place in responses_dir
        gzreduction.panoptes.api.api_to_json.get_latest_classifications(
            responses_dir,  #
            previous_dir=None,
            max_classifications=1e8,
            manual_last_id=self.last_id  # should be the last id used in the last major export (provided to shards)
        )
        raw_responses_loc = gzreduction.panoptes.api.api_to_json.get_chunk_files(responses_dir, derived=False)[0]
        gzreduction.panoptes.api.reformat_api_like_exports.derive_chunk(raw_responses_loc)
        derived_responses_loc = gzreduction.panoptes.api.api_to_json.get_chunk_files(responses_dir, derived=True)[0]

        preprocessed_dir = '/tmp/preprocessed_classifications'
        if os.path.isdir(preprocessed_dir):
            shutil.rmtree(preprocessed_dir)
        else:
            os.mkdir(preprocessed_dir)
    
        # make flat table of classifications. Basic use-agnostic view. 
        gzreduction.panoptes.panoptes_to_responses.preprocess_classifications(
            [derived_responses_loc],
            gzreduction.schemas.dr5_schema,
            start_date=datetime(year=2018, month=3, day=15),  # public launch  tzinfo=timezone.utc
            save_dir=preprocessed_dir
        )

        votes_loc = os.path.join(preprocessed_dir, 'latest_votes.csv')
        predictions_loc = os.path.join(preprocessed_dir, 'latest_predictions.csv')
        
        responses = gzreduction.panoptes.responses_to_votes.join_response_shards(preprocessed_dir)
        gzreduction.panoptes.responses_to_votes.get_votes(
            responses,
            question_col='task', 
            answer_col='value', 
            schema=gzreduction.schemas.dr5_schema,
            save_loc=votes_loc)

        panoptes_votes = pd.read_csv(votes_loc)
        logging.debug(panoptes_votes['value'].value_counts())

        gzreduction.votes_to_predictions.votes_to_predictions(
            panoptes_votes,
            gzreduction.schemas.dr5_schema,
            reduced_save_loc=None,
            predictions_save_loc=predictions_loc
    )

        tweaked_predictions_loc = os.path.join(preprocessed_dir, 'tweaked_predictions.csv')
        prepare_catalogs.tweak_previous_decals_classifications(predictions_loc, tweaked_predictions_loc)

        classifications = pd.read_csv(tweaked_predictions_loc)
        classifications = create_experiment_catalog.filter_classifications(classifications, self.question)  # for retired
        classifications = create_experiment_catalog.define_labels(classifications, self.question)
        return classifications['id_str'].values, classifications['label'].values, classifications['total_votes'].values
        

    def save(self, save_dir):
        data = {
            'catalog_loc': self._catalog_loc,
            'login_loc': self._login_loc,
            'project_id': self._project_id
        }
        with open(os.path.join(save_dir, 'oracle_config.json'), 'w') as f:
            json.dump(data, f)

def load_panoptes_oracle(save_dir):
    with open(os.path.join(save_dir, 'oracle_config.json'), 'r') as f:
        return Panoptes(**json.load(f))

class PanoptesMock(Oracle):

    def __init__(self, oracle_loc, subjects_requested_loc):
        assert os.path.isfile(oracle_loc)  # must already exist
        logging.info('Using oracle loc: {}'.format(oracle_loc))
        logging.info('Using subjects requested loc: {}'.format(subjects_requested_loc))
        self._oracle_loc = oracle_loc
        self._subjects_requested_loc = subjects_requested_loc

    def request_labels(self, subject_ids, name):
        logging.info('Pretending to upload {} subjects: {}'.format(len(subject_ids), name))
        assert len(set(subject_ids)) == len(subject_ids)  # must be unique
        with open(self._subjects_requested_loc, 'w') as f:
            json.dump(subject_ids, f)

    def get_labels(self):
        # oracle.csv is created by make_shards.py, contains label and id_str pairs of vote fractions
        if not os.path.isfile(self._subjects_requested_loc):
            logging.warning(
                'No previous subjects requested at {}'.format(self._subjects_requested_loc))
            return [], [], []  # must unpack 3 values, look here if 'not enough values to unpack' error

        with open(self._subjects_requested_loc, 'r') as f:
            subject_ids = json.load(f)
        assert isinstance(subject_ids, list)
        assert len(subject_ids) > 0
        assert len(set(subject_ids)) == len(subject_ids)  # must be unique
        os.remove(self._subjects_requested_loc)

        known_catalog = pd.read_csv(
            self._oracle_loc,
            usecols=['id_str', 'label', 'total_votes'],
            dtype={'id_str': str, 'label': int, 'total_votes': int}
        )
        # return labels from the oracle, mimicking live GZ classifications
        labels = []
        id_str_dummy_df = pd.DataFrame(data={'id_str': subject_ids})
        matching_df = pd.merge(id_str_dummy_df, known_catalog, how='inner', on='id_str')
        labels = list(matching_df['label'].astype(int))
        total_votes = list(matching_df['total_votes'].astype(int))
        assert len(id_str_dummy_df) == len(matching_df)
        assert len(subject_ids) == len(labels)
        return subject_ids, labels, total_votes

    def save(self, save_dir):
        data = {
            'oracle_loc': self._oracle_loc,
            'subjects_requested_loc': self._subjects_requested_loc,
        }
        with open(os.path.join(save_dir, 'oracle_config.json'), 'w') as f:
            json.dump(data, f)

def load_panoptes_mock_oracle(save_dir):
    with open(os.path.join(save_dir, 'oracle_config.json'), 'r') as f:
        return PanoptesMock(**json.load(f))

def load_oracle(save_dir):
    try:
        return load_panoptes_oracle(save_dir)
    except KeyError:  # TODO actually wrong exception
        return load_panoptes_mock_oracle(save_dir)
