
import os
import json
import requests
import pandas as pd

from panoptes_client import Panoptes, Workflow, SubjectSet
from zoobot.active_learning import mock_panoptes

from gzreduction.panoptes.api import api_to_json, reformat_api_like_exports


def configure_designator(subject_set_chances, workflow_id, login_loc, debug=False):
    if debug:
        os.environ['PANOPTES_DEBUG'] = 'True'

    with open(login_loc, 'r') as f:
        zooniverse_login = json.load(f)
    Panoptes.connect(**zooniverse_login)

    workflow = Workflow.find(workflow_id)
    workflow.configuration['subject_set_chances'] = subject_set_chances
    workflow.configuration = workflow.configuration
    workflow.save()


def upload_dummy_subjects(catalog_loc, project_id, workflow_id, login_loc):
    # only for testing 

    catalog = pd.read_csv(catalog_loc, nrows=10000)

    dummy_catalog_loc = 'data/gz2/master_catalog_pretend_ec2.csv'
    catalog['file_loc'] = catalog['local_png_loc']
    catalog.to_csv(dummy_catalog_loc, index=False)

    featured_col = 'smooth-or-featured_featured-or-disk'
    most_featured = catalog.sort_values(featured_col, ascending=False)[:80]['id_str'].values
    least_featured = catalog.sort_values(featured_col, ascending=True)[:500]['id_str'].values

    panoptes = mock_panoptes.Panoptes(
        catalog_loc=dummy_catalog_loc,
        login_loc=login_loc,
        project_id=project_id,
        # don't worry about these for now, not getting labels
        workflow_id=None,
        last_id=None,
        question=None
    )

    panoptes.request_labels(most_featured, name='priority', retirement=5)
    panoptes.request_labels(least_featured, name='random', retirement=1)


if __name__ == '__main__':

    project_id = '8751'
    workflow_id = '10582'
    login_loc = 'zooniverse_login.json'
    catalog_loc = 'data/decals/decals_master_catalog.csv'

    # upload_dummy_subjects(catalog_loc, project_id, workflow_id, login_loc)

    subject_set_chances = {
            '74909': 0.8,  # priority
            '74905': 0.2  # random
    }
    configure_designator(subject_set_chances, workflow_id, login_loc, debug=True)


