import numpy as np

import openpi_client.image_tools as image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = np.zeros((1, 50, 50, 3), dtype=np.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = np.zeros((1, 256, 320, 3), dtype=np.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)


def test_jpeg_roundtrip_shrinks_payload_and_preserves_image():
    """Wire compression: encode/decode must round-trip and actually shrink."""
    import numpy as np
    from openpi_client import image_tools, msgpack_numpy

    # Smooth gradient plus a block -- representative of a rendered scene rather
    # than random noise, which is JPEG's pathological worst case.
    y, x = np.mgrid[0:224, 0:224]
    img = np.stack([x * 255 // 224, y * 255 // 224, (x + y) * 255 // 448], -1).astype(np.uint8)
    img[60:150, 60:150] = [200, 30, 30]

    raw = msgpack_numpy.packb({"a": img, "b": img})
    enc = msgpack_numpy.packb({"a": image_tools.encode_jpeg(img), "b": image_tools.encode_jpeg(img)})
    assert len(enc) * 5 < len(raw), f"expected >5x shrink, got {len(raw) / len(enc):.1f}x"

    out = image_tools.decode_tree(msgpack_numpy.unpackb(enc))
    assert out["a"].shape == img.shape
    assert out["a"].dtype == np.uint8
    # Lossy, so compare on mean error rather than exact equality.
    assert np.abs(out["a"].astype(int) - img.astype(int)).mean() < 3.0


def test_decode_tree_passes_through_uncompressed_requests():
    """A request with no JPEG marker must be returned untouched (old clients)."""
    import numpy as np
    from openpi_client import image_tools

    obs = {"observation/img": np.zeros((4, 4, 3), np.uint8), "prompt": "pick it up", "n": 3}
    out = image_tools.decode_tree(obs)
    assert np.array_equal(out["observation/img"], obs["observation/img"])
    assert out["prompt"] == "pick it up"
    assert out["n"] == 3
