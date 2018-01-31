

import pandas as pd

from zoobot.shared_utilities import match_galaxies_to_catalog


def get_classification_results(published_data_loc, nrows=None):
    """
    Get classification data table from GZ2 published results.
    Get the weighted fraction and raw count data for each relevant question.
    Currently, this is for the questions smooth/featured, edge-on, round/cigar, spiral/not, and spiral count.
    https://data.galaxyzoo.org/ Galaxy Zoo 2 Table 1
    Direct link: http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz

    Args:
        published_data_loc (str): file location of published GZ classifications table
        nrows (int): num. of rows to return (from head). If None, return all rows.

    Returns:
        (pd.DataFrame) DR7 ID, spiral weighted vote fraction, and vote counts for spiral/not spiral, by subject
    """

    relevant_answers = [
        't01_smooth_or_features_a01_smooth',
        't01_smooth_or_features_a02_features_or_disk',
        't01_smooth_or_features_a03_star_or_artifact',

        't02_edgeon_a04_yes',
        't02_edgeon_a05_no',

        't04_spiral_a08_spiral',
        't04_spiral_a09_no_spiral',

        't07_rounded_a16_completely_round',
        't07_rounded_a17_in_between',
        't07_rounded_a18_cigar_shaped',

        't11_arms_number_a31_1',
        't11_arms_number_a32_2',
        't11_arms_number_a33_3',
        't11_arms_number_a34_4',
        't11_arms_number_a36_more_than_4',
        't11_arms_number_a37_cant_tell'
    ]

    relevant_values = [
        '_weighted_fraction',
        '_count'
    ]

    useful_columns = ['dr7objid', 'ra', 'dec']
    for answer in relevant_answers:
        for value in relevant_values:
            useful_columns.append("".join([answer, value]))

    df = pd.read_csv(published_data_loc, nrows=nrows, usecols=useful_columns)

    return df


# def get_catalog(published_data_loc, subject_manifest_loc, labels_loc, nrows=None, plot_overlap=False):
#     """
#     Load published data (Hart 2016) and subject manifest for AWS. Match on RA/DEC.
#     Get the weighted fraction and raw count data for each relevant question.
#     Currently, this is for the questions smooth/featured, edge-on, round/cigar, spiral/not, and spiral count.
#     Optionally, plot the overlap in RA/DEC of both catalogs
#     Args:
#         published_data_loc (str): file location of Hart 2016 GZ2 results catalog
#         subject_manifest_loc (str): file location of AWS subject manifest, private comm. from Sandor Kruk
#         labels_loc (str): file location to matched catalog
#         nrows (int): max number of rows to load per catalog, for speedy debugging. if None, load all rows.
#         plot_overlap (bool): if True, plot overlap of both catalogs in RA/DEC
#
#     Returns:
#         (pd.DataFrame) matched catalog of GZ2 classifications and AWS locations, with answers to relevant questions
#     """
#     classifications = get_classification_results(published_data_loc, nrows=nrows)
#     print('Published subjects with labels: {}'.format(len(classifications)))
#
#     subject_manifest = pd.read_csv(subject_manifest_loc, nrows=nrows)
#     print('AWS subjects from Sandor: {}'.format(len(subject_manifest)))
#
#     # TODO I don't know how to install datashader on Travis
#     # if plot_overlap:
#     #     plot_catalog_overlap(
#     #         classifications,
#     #         subject_manifest,
#     #         ['published subjects', 'sandor AWS subjects'],
#     #         'sandor_and_volunteers_overlap')
#
#     catalog = match_galaxies_to_catalog(classifications, subject_manifest)
#     assert len(catalog) > 0
#
#     catalog.to_csv(labels_loc)
#     return catalog
