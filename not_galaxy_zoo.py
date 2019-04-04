

import json
import requests
import pandas as pd

from panoptes_client import Panoptes, Workflow, SubjectSet
from zoobot.active_learning import mock_panoptes

from gzreduction.panoptes.api import api_to_json, reformat_api_like_exports

if __name__ == '__main__':


    # auth_token = reformat_api_like_exports.get_panoptes_auth_token()
    # headers = reformat_api_like_exports.get_panoptes_headers(auth_token)
    # url = 'https://panoptes.zooniverse.org/api/workflows/{}'.format(workflow_id)
    # response = requests.get(url, headers=headers)
    # current_config = response.json()['workflows'][0]['configuration']
    # print(current_config)


    # # dict of form: {'workflows': [{'id': '9816', 'display_name': ... }

    # updated_workflow = current_workflow.copy()
    # raw_response = requests.put(url, data=updated_workflow, headers=headers)
    # print(raw_response)


    # response = raw_response.json()
    # print(response)
    # print(response.json())
    
    # print('done')




    # updated_workflow['workflows'][0]['configuration']['subject_set_chances'] = subject_set_chances

    # project_id = ''

    # print(current_workflow['workflows'][0]['configuration'])

    login_loc = 'zooniverse_login.json'
    workflow_id = '9816'
    catalog_loc = 'data/gz2/master_catalog.csv'

    catalog = pd.read_csv(catalog_loc, nrows=10000)

    dummy_catalog_loc = 'data/gz2/master_catalog_pretend_ec2.csv'
    catalog['file_loc'] = catalog['local_png_loc']
    catalog.to_csv(dummy_catalog_loc, index=False)

    featured_col = 'smooth-or-featured_featured-or-disk'
    most_featured = catalog.sort_values(featured_col, ascending=False)[:80]['id_str'].values
    least_featured = catalog.sort_values(featured_col, ascending=True)[:500]['id_str'].values

    # TODO should modify existing sets, not create new ones
    panoptes = mock_panoptes.Panoptes(
        catalog_loc=dummy_catalog_loc,
        login_loc=login_loc,
        project_id='8751',
        # don't worry about these for now, not getting labels
        last_id=None,
        question=None
    )

    panoptes.request_labels(most_featured, name='priority', retirement=5)
    panoptes.request_labels(least_featured, name='normal', retirement=1)

    # import os
    # os.environ['PANOPTES_DEBUG'] = 'True'

    # with open(login_loc, 'r') as f:
    #     zooniverse_login = json.load(f)
    # Panoptes.connect(**zooniverse_login)

    # subject_set_chances = {
    #     '74577': 0.99,  # priority
    #     '74578': 0.01  # normal
    # }

    # workflow = Workflow.find(workflow_id)
    # print(workflow.configuration)
    # workflow.configuration['subject_set_chances'] = subject_set_chances
    # workflow.configuration = workflow.configuration
    # workflow.save()

    # retired_subjects_id = '73892'
    # subjects = SubjectSet.find(retired_subjects_id)
    # gen = subjects.subjects
    # for n in range(10):
    #     subject = next(gen)



# auth_token = reformat_api_like_exports.get_panoptes_auth_token()
# headers = reformat_api_like_exports.get_panoptes_headers(auth_token)
# url = 'https://panoptes.zooniverse.org/api/SubjectWorkflowStatuses?workflow_id={}'.format(workflow_id)
# response = requests.get(url, headers=headers)
# print(response)
# print(response.json())



"""
I just realised that I also need to get the subjects for a workflow, in order to be able to match the classifications to the galaxy catalog.

classification <-(.links.subject, .id)-> subject <-(.metadata.galaxy_id, .galaxy_id)-> galaxy_catalog_entry 

I think I can do this with https://panoptes.docs.apiary.io/#reference/subjects/subject/retrieve-a-single-subject
"""
