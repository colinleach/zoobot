from tensorflow.contrib import predictor
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord

columns_to_save = [
    'smooth-or-featured_smooth',
    'smooth-or-featured_featured-or-disk',
    'smooth-or-featured_artifact',
    'smooth-or-featured_total-votes',
    'smooth-or-featured_smooth_fraction',
    'smooth-or-featured_featured-or-disk_fraction',
    'smooth-or-featured_artifact_fraction',
    'smooth-or-featured_smooth_min',
    'smooth-or-featured_smooth_max',
    'smooth-or-featured_featured-or-disk_min',
    'smooth-or-featured_featured-or-disk_max',
    'smooth-or-featured_artifact_min',
    'smooth-or-featured_artifact_max',
    'smooth-or-featured_prediction-encoded',  # 0 for artifact, 1 for featured, 2 for smooth
    'classifications_count',
    # 'iauname', string features not yet supported
    'subject_id',
    'nsa_id',
    'ra',
    'dec']

df_loc = '/data/repos/galaxy-zoo-panoptes/reduction/data/output/panoptes_predictions_with_catalog.csv'
df = pd.read_csv(df_loc, usecols=columns_to_save + ['fits_loc', 'png_loc', 'png_ready'], dtype={'fits_loc': str}, nrows=50)

# df['fits_loc'] = df['fits_loc'].apply(lambda x: x[2:-1])
# temporarily correct for weird bytes conversion from reduction


def serialise_row(row):
    return catalog_to_tfrecord.row_to_serialized_example(
        row,
        img_size=28,
        label_col='smooth-or-featured_prediction-encoded',
        columns_to_save=columns_to_save,
        source='fits')


galaxies_to_predict = df[:1].apply(serialise_row, axis=1)

# example = read_tfrecord.read_and_decode_single_example(test_tfrecord_loc)

export_dir = '/Data/repos/zoobot/zoobot/runs/chollet_panoptes_featured_bayesian_temp28/1526488263'
predict_fn = predictor.from_saved_model(export_dir)
predictions = predict_fn({'examples': galaxies_to_predict})
print(predictions['featured_score'])
# print(predictions['logits'])
