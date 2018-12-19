from zoobot.illustris import illustris_to_tfrecord


def test_render_fits(subject):
    pil_img = illustris_to_tfrecord.load_illustris_as_pil(subject)
    # pil_img.show()


def test_remove_mergers_from_random_sample(catalog):
    clean_catalog = illustris_to_tfrecord.remove_known_galaxies_from_random_sample(catalog)
    assert all(clean_catalog['id'].values == ['0', '1', '2', '3', '4', '5', '9'])
