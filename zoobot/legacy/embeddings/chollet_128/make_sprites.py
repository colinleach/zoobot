import numpy as np


def images_to_sprite(images, labels=None):
    """Creates the sprite image along with any necessary padding

    Args:
      images: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """

    images = images[:, :, :, 0]  # greyscale

    # re-order batch dimension by label
    if labels is not None:
        images = sort_images_by_labels(images, labels)

    if len(images.shape) == 3:
        images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))  # add a third dimension
    min = np.min(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    # add padding to one thumbnail
    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant',
                    constant_values=0)

    # Tile the individual thumbnails into an image.
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3)
                                                                 + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])

    return images


def sort_images_by_labels(images, labels):
    sort_index = np.argsort(labels)
    sorted_images = images[sort_index, :, :]
    return sorted_images
