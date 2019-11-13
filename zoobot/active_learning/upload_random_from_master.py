import os
import tempfile

import pandas as pd

from zoobot.active_learning import mock_panoptes

if __name__ == '__main__':

    # careful, may technically be a different master catalog
    master_catalog_loc = 'data/decals/decals_master_catalog.csv'
    login_loc = 'zooniverse_login.json'

    project_id = '5733'  # main GZ project
    # project_id = '6490'  # mobile GZ project

    df = pd.read_csv(master_catalog_loc)
    unlabelled = df[pd.isnull(df['smooth-or-featured_total-votes'])]
    unlabelled['id_str'] = unlabelled['iauname']  # my client expects this column
    print('{} of {} unlabelled'.format(len(unlabelled), len(df)))

    unlabelled['file_loc'] = unlabelled['local_png_loc']
    print(unlabelled['file_loc'][:5])

    # touch table will work backwards through the list
    # unlabelled = unlabelled.sort_values('file_loc', ascending=False)
    # selected = slice(20000, 25000)
    # name = '2019-06-14_touch_table_5k'
    # retirement = 40

    # galaxy zoo (and mobile app) will work forwards
    unlabelled = unlabelled.sort_values('file_loc')
    selected = slice(65000, 75000)
    name = 'random'
    retirement = 3

    with tempfile.TemporaryDirectory() as tempdir:
        unlabelled_loc = os.path.join(tempdir, 'unlabelled.csv')

        unlabelled[selected].to_csv(unlabelled_loc)
        panoptes = mock_panoptes.Panoptes(
            catalog_loc=unlabelled_loc,
            login_loc=login_loc,
            project_id=project_id,
            # don't worry about these, not getting labels
            workflow_ids=[],
            last_id=None,
            question=None
        )
    panoptes.request_labels(unlabelled['id_str'].values, name=name, retirement=retirement)
