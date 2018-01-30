import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymongo import MongoClient

from zoobot.shared_utilities import plot_catalog

pd.options.display.max_rows = 200
pd.options.display.max_columns = 100
sns.set_context('notebook')


def create_subject_manifest(subject_manifest_loc, survey_tag, max_subjects=None):
    """
    Convert Galaxy Zoo 2 classifications export into subject manifest
    subject manifest is list of subjects with id, ra, dec

    Args:
        subject_manifest_loc (str): absolute path to save subject manifest
        survey_tag (str): Survey tag to filter subjects by e.g. 'sloan'. For all subjects, use 'all'

    Returns:
        None
    """

    client = MongoClient()
    db = client.galaxy_zoo

    subjects = db.subjects

    if survey_tag == 'all':
        cursor = subjects.find()
    else:
        cursor = subjects.find({'metadata.survey': survey_tag})

    sloan_data = []
    for document in cursor[:max_subjects]:
        galaxy = {
            'zooniverse_id': document['zooniverse_id'],
            'ra': document['coords'][0],
            'dec': document['coords'][1],
            'img_server_loc': document['location']['standard'],
            'created_at': document['created_at'],
            'activated_at': document['activated_at'],
            'classification_count': document['classification_count'],
            'petrorad_50_r': document['metadata']['petrorad_50_r'],
            'r_mag': document['metadata']['mag']['r'],
            'abs_r_mag': document['metadata']['mag']['abs_r']
        }
        sloan_data.append(galaxy)

    df = pd.DataFrame(sloan_data)

    # round dates to the calendar date, but keep as datetime objects
    df['created_at_date'] = pd.to_datetime(df['created_at'].dt.date)
    df['activated_at_date'] = pd.to_datetime(df['activated_at'].dt.date)

    df.index.name = 'index'
    df.to_csv('/data/galaxy_zoo/gz2/subjects/{}_subjects.csv'.format(survey_tag))


if __name__ == '__main__':

    nrows = None
    survey_tag = 'sloan'
    force_new = False

    catalog_dir = '/data/galaxy_zoo/gz2/subjects'
    subject_manifest_loc = '{}/{}_subjects.csv'.format(catalog_dir, survey_tag)

    if not os.path.exists(subject_manifest_loc) or force_new:
        # requires mongodb server - open terminal and run 'mongod'. Use ctrl-c to shut down.
        create_subject_manifest(subject_manifest_loc, survey_tag, max_subjects=None)  # requires mongodb. Run mongod

    df = pd.read_csv(subject_manifest_loc)

    print(len(df))
    df = df[df['abs_r_mag'] > -50]
    print(len(df))

    print(len(df))
    df = df[df['petrorad_50_r'] < 40]
    print(len(df))

    print(df['abs_r_mag'].max())

    print(df['created_at_date'].value_counts())

    activated_at_date = df['activated_at_date'].value_counts().sort_index().reset_index()
    activated_at_date.rename(
        inplace=True,
        columns={
            'index': 'date',
            'activated_at_date': 'subjects_activated'}
    )
    activated_at_date.set_index('date', drop=True, inplace=True)
    # needs a datetimeindex as index, rather than datetime values
    activated_at_date.index = pd.DatetimeIndex(activated_at_date.index)

    index = pd.date_range(start=df['activated_at_date'].min(), end=df['activated_at_date'].max())

    print(activated_at_date)
    print(index)

    activated_at_date = activated_at_date.reindex(index, fill_value=0)
    print(activated_at_date)
    print(activated_at_date['subjects_activated'].value_counts())
    # exit(0)

    plt.plot(pd.to_datetime(activated_at_date.index.values), activated_at_date['subjects_activated'])
    # plt.hist(pd.to_datetime(df['activated_at_date']), np.ones(len(df)))
    fig = plt.gcf().autofmt_xdate()
    plt.xlabel('date')
    plt.ylabel('{} subjects activated'.format(survey_tag))
    plt.tight_layout()
    plt.savefig('figures/{}_activation_dates.png'.format(survey_tag))

    plt.clf()
    df.hist('r_mag', bins=50)
    plt.title('')
    plt.xlabel('r mag')
    plt.ylabel('{} subjects'.format(survey_tag))
    plt.tight_layout()
    plt.savefig('figures/{}_r_mag.png'.format(survey_tag))

    plt.clf()
    df.hist('abs_r_mag', bins=50)
    plt.title('')
    plt.xlabel('absolute r mag')
    plt.ylabel('{} subjects'.format(survey_tag))
    plt.tight_layout()
    plt.savefig('figures/{}_abs_r_mag.png'.format(survey_tag))

    plt.clf()
    df.hist('petrorad_50_r', bins=50)
    plt.title('')
    plt.xlabel('petro50 radius')
    plt.ylabel('{} subjects'.format(survey_tag))
    plt.tight_layout()
    plt.savefig('figures/{}_petro50.png'.format(survey_tag))

    plt.clf()
    df.hist('classification_count', bins=50)
    plt.title('')
    plt.xlabel('classification count')
    plt.ylabel('{} subjects'.format(survey_tag))
    plt.tight_layout()
    plt.savefig('figures/{}_classification_count.png'.format(survey_tag))

    # datashader does not support filetypes only png
    # plot_catalog(df, 'figures/{}_subjects_from_mongodb'.format(survey_tag), column_to_colour='abs_r_mag')
    plot_catalog(df, 'figures/{}_subjects_from_mongodb'.format(survey_tag))

    plot_catalog(df[df['abs_r_mag'] > -17.7], 'figures/{}_subjects_from_mongodb_bright'.format(survey_tag))
    plot_catalog(df[df['abs_r_mag'] < -17.7], 'figures/{}_subjects_from_mongodb_faint'.format(survey_tag))
