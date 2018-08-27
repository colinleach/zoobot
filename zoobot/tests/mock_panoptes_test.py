import pytest

from zoobot.active_learning import mock_panoptes


def test_get_labels():
    subject_ids = ['20530826', '20531460']
    labels = mock_panoptes.get_labels(subject_ids)
    assert labels == [1, 1]