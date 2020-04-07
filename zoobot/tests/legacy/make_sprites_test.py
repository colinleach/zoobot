# import pytest

# from zoobot.embeddings.chollet_128.make_sprites import *


# @pytest.fixture()
# def labels():
#     return [0., 1., 0., 0.]


# @pytest.fixture()
# def images():
#     images = np.ones((4, 3, 3, 1), dtype=float) * 0.1
#     images[1, :, :, :] = np.ones((3, 3, 1), dtype=float) * 0.7
#     return images


# def test_sort_images_by_label(images, labels):
#     sorted_images = sort_images_by_labels(images, labels)
#     assert sorted_images[0, :, :, :].max() == 0.1
#     assert sorted_images[3, :, :, :].max() == 0.7


# def test_images_to_sprites(images):
#     sprite = images_to_sprite(images)
#     print(sprite)


# # def test_images_to_sprites_sorted(images, labels):
# #     sprite = images_to_sprite(images, labels)
# #     print(sprite)