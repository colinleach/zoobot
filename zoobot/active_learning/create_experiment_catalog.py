import os

import pandas as pd

"""Shared logic to refine catalogs"""

def finalise_catalog(catalog, question, save_dir):
    catalog = shuffle(catalog)  # crucial for GZ2!
    catalog = filter_classifications(catalog, question)
    catalog = define_labels(catalog, question)
    catalog = specify_file_locs(catalog)
    make_oracle(catalog, save_dir)
    return catalog


def filter_classifications(catalog, question):
    if question == 'smooth':  # artificially enforce as simple test case
        catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    elif question == 'bar':  # filter to at least a bit featured
        catalog = catalog[catalog['bar_total-votes'] > 10]  
    else:
        raise ValueError('question {} not understood'.format(question))
    return catalog


def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()


def define_labels(catalog, question):
    catalog['total_votes'] = catalog['smooth-or-featured_total-votes']
    if question == 'smooth':
        # print(catalog.columns.values)
        catalog['label'] = catalog['smooth-or-featured_smooth']
    elif question == 'bar':
        catalog['total_votes'] = catalog['bar_total-votes']
        try:
            catalog['label'] = catalog['bar_weak']  # DECALS
        except KeyError:
            catalog['label'] = catalog['bar_yes']  # GZ2
    else:
        raise ValueError('question {} not understood'.format(question))
    return catalog


def specify_file_locs(catalog):
    catalog['file_loc'] = '/root/repos/zoobot/' + catalog['png_loc']  # now expects this to point to png loc relative to repo root
    assert all(loc for loc in catalog['file_loc'])
    del catalog['png_loc']  # else may load this by default
    print(catalog['file_loc'].sample(5))
    return catalog


def make_oracle(catalog, save_dir):  # move out of shard dir - allow oracle to be in a different directory, probably experiment directory
    # save catalog for PanoptesMock to return
    catalog[['id_str', 'total_votes', 'label']].to_csv(os.path.join(save_dir, 'oracle.csv'), index=False)
    catalog.to_csv(os.path.join(save_dir, 'full_catalog.csv'), index=False)


def make_mock_catalogs(catalog, train_size=256, eval_size=2500):
    # given a (historical) catalog, pretend split into labelled and unlabelled
    labelled_size = train_size + eval_size
    labelled_catalog = catalog[:labelled_size]  # for training and eval. Could do basic split on these!
    unlabelled_catalog = catalog[labelled_size:]  # for pool
    del unlabelled_catalog['label']

    return labelled_catalog, unlabelled_catalog


if __name__ == '__main__':

    # apply filters and specify the question to solve

    catalog_loc = '/data/repos/gzreduction/data/predictions/example_panoptes_predictions.csv'
    catalog_dir = 'data/decals/prepared_catalogs/smooth_unfiltered'
    if not os.path.isdir(catalog_dir):
        os.mkdir(catalog_dir)

    catalog = pd.read_csv(catalog_loc)
    catalog = finalise_catalog(catalog, 'smooth', catalog_dir)

    labelled_catalog, unlabelled_catalog = make_mock_catalogs(catalog)
    labelled_catalog.to_csv(os.path.join(catalog_dir, 'labelled_catalog.csv'), index=False)
    unlabelled_catalog.to_csv(os.path.join(catalog_dir, 'unlabelled_catalog.csv'), index=False)
