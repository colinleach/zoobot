from zoobot.tests import TEST_EXAMPLE_DIR


@pytest.fixture()
def example_tfrecord():
    return {
        'loc': TEST_EXAMPLE_DIR + '/panoptes_featured_s128_l0.4_test.tfrecord',
        'size': 28,
        'channels': 3


def example_galaxy()
