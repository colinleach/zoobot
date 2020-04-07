import os

from zoobot.tests.active_learning import conftest  # to patch
from zoobot import estimators  # to patch
from zoobot.active_learning import database, db_access
from zoobot.tfrecord import read_tfrecord

def test_write_catalog_to_tfrecord_shards(unlabelled_catalog, empty_shard_db, size, channels, tfrecord_dir):
    columns_to_save = ['id_str', 'some_feature']
    database.write_catalog_to_tfrecord_shards(
        unlabelled_catalog,
        empty_shard_db,
        size,
        columns_to_save,
        tfrecord_dir,
        shard_size=15)
    # verify_db_matches_catalog(catalog, empty_shard_db, 'id_str', label_col)
    database.verify_db_matches_shards(empty_shard_db, size, channels)
    database.verify_catalog_matches_shards(unlabelled_catalog, empty_shard_db, size, channels)


def test_make_predictions_on_tfrecord(monkeypatch, tfrecord_matrix_id_loc, filled_shard_db, size):
    monkeypatch.setattr(
        estimators.make_predictions,
        'get_samples_of_images',
        conftest.mock_get_samples_of_images
    )

    MAX_ID_STR = 64
    def mock_subject_is_labelled(id_str, db):
        return int(id_str) <= MAX_ID_STR
    monkeypatch.setattr(
        db_access,
        'subject_is_labelled',
        mock_subject_is_labelled
    )

    n_samples = 10
    model = None  # avoid this via mocking, above
    subjects, samples = database.make_predictions_on_tfrecord(
        [tfrecord_matrix_id_loc],
        model,  
        filled_shard_db,
        n_samples=n_samples,
        size=size,
        max_images=20000
    )
    assert samples.shape == (len(subjects), n_samples)
    assert [int(subject['id_str']) > MAX_ID_STR for subject in subjects]  # no labelled subjects 



def test_add_labelled_subjects_to_tfrecord(filled_shard_db_with_labels, tfrecord_dir, size, channels):
    tfrecord_loc = os.path.join(tfrecord_dir, 'active_train.tfrecord')  # place images here
    subject_ids = ['some_hash', 'yet_another_hash']
    database.add_labelled_subjects_to_tfrecord(filled_shard_db_with_labels, subject_ids, tfrecord_loc, size, ['id_str', 'column'])

    # open up the new record and check
    subjects = read_tfrecord.load_examples_from_tfrecord([tfrecord_loc], read_tfrecord.matrix_id_feature_spec(size, channels))
    # should NOT be read back shuffled!
    assert subjects[0]['id_str'] == 'some_hash'.encode('utf-8')  # tfrecord saves as bytes
    assert subjects[1]['id_str'] == 'yet_another_hash'.encode('utf-8')  #tfrecord saves as bytes



def test_filter_for_new_only(filled_shard_db_with_partial_labels):
    all_subject_ids = ['some_hash', 'some_other_hash']  
    all_labels = [{'column': 1}, {'column': 4}]

    subject_ids, labels = database.filter_for_new_only(
        filled_shard_db_with_partial_labels, # some_hash already has a label in this db
        all_subject_ids,
        all_labels
    )
    assert subject_ids == ['some_other_hash']
    assert labels == [{'column': 4}]

def test_db_fully_labelled_empty(filled_shard_db):
    assert not database.db_fully_labelled(filled_shard_db)

def test_db_fully_labelled_partial(filled_shard_db_with_partial_labels):
    assert not database.db_fully_labelled(filled_shard_db_with_partial_labels)

def test_db_fully_labelled_full(filled_shard_db_with_labels):
    assert database.db_fully_labelled(filled_shard_db_with_labels)
