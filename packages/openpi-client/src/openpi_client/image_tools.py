import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


# --- wire compression -------------------------------------------------------
# Observations cross an SSH tunnel to the cluster as raw uint8 arrays: two
# 224x224x3 images msgpack to 294 KB per request. Measured on a 16-env RoboLab
# sweep, transfer was ~173 ms of a ~560 ms round trip. JPEG at q=95 cuts the
# payload ~12x, so the tunnel stops being a meaningful share of it.
#
# The encoding is DELIBERATELY explicit rather than automatic: images are wrapped
# in a marker dict so the server can tell an encoded image from a plain array and
# decode only what was encoded. Old clients keep working unchanged -- a request
# with no marker is passed straight through.
#
# Note JPEG is lossy. q=95 is visually indistinguishable and DROID's own training
# frames are JPEG-compressed video, so the artifacts are in distribution -- but
# it is NOT bit-identical, so any adoption must be equivalence-tested on success
# rate rather than assumed.

_JPEG_KEY = "__jpeg__"


def encode_jpeg(img: np.ndarray, quality: int = 95) -> dict:
    """Wrap a HxWx3 uint8 image as a JPEG-encoded marker dict for the wire."""
    import io

    buf = io.BytesIO()
    Image.fromarray(convert_to_uint8(img)).save(buf, format="JPEG", quality=quality)
    return {_JPEG_KEY: buf.getvalue()}


def decode_jpeg(value):
    """Inverse of `encode_jpeg`. Anything that is not a marker dict passes through."""
    import io

    if isinstance(value, dict) and _JPEG_KEY in value:
        return np.asarray(Image.open(io.BytesIO(value[_JPEG_KEY])).convert("RGB"))
    return value


def decode_tree(obs: dict) -> dict:
    """Decode every JPEG-marked entry in an observation dict; leave the rest alone."""
    if not isinstance(obs, dict):
        return obs
    return {k: decode_jpeg(v) for k, v in obs.items()}
