import pytest

from zoobot.illustris import illustris_to_tfrecord


@pytest.fixture()
def fits_loc():
    return 'zoobot/test_examples/illustris_test_dir/synthetic_image_104798_band_5_camera_0_bg_1.fits'


@pytest.fixture()
def subject(fits_loc):
    return {'fits_loc': fits_loc}


def test_render_fits(subject):
    pil_img = illustris_to_tfrecord.load_illustris_as_pil(subject)
    # pil_img.show()
