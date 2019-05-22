import os
import tempfile

import pandas as pd

from zoobot.active_learning import mock_panoptes

if __name__ == '__main__':

    # careful, may technically be a different master catalog
    master_catalog_loc = 'data/decals/decals_master_catalog.csv'
    login_loc = 'zooniverse_login.json'
    project_id = '5733'

    df = pd.read_csv(master_catalog_loc)
    unlabelled = df[pd.isnull(df['smooth-or-featured_total-votes'])]
    unlabelled['id_str'] = unlabelled['iauname']  # my client expects this column
    print('{} of {} unlabelled'.format(len(unlabelled), len(df)))

    unlabelled['file_loc'] = unlabelled['local_png_loc']

    # selected = slice(0, 5000)
    selected = slice(7000, 8000)
    with tempfile.TemporaryDirectory() as tempdir:
        unlabelled_loc = os.path.join(tempdir, 'unlabelled.csv')
        unlabelled[selected].to_csv(unlabelled_loc)
        panoptes = mock_panoptes.Panoptes(
            catalog_loc=unlabelled_loc,
            login_loc=login_loc,
            project_id=project_id,
            # don't worry about these, not getting labels
            workflow_id=None,
            last_id=None,
            question=None
        )
    panoptes.request_labels(unlabelled['id_str'].values, name='random', retirement=3)
