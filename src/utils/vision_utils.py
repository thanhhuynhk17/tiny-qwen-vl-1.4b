from PIL import Image
def add_image_padding(image: Image.Image, keep_ratio: bool = True) -> Image.Image:
    """
    Add padding to a PIL image to make it square while keeping the aspect ratio.

    Converts the image to RGB mode if not already.

    Args:
        image (PIL.Image.Image): The input image.
        keep_ratio (bool): Whether to keep the aspect ratio. If False, the image will be returned as it is.

    Returns:
        PIL.Image.Image: The padded or resized RGB image.
    """
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    w, h = image.size

    # Skip if already square
    if w == h:
        return image

    if keep_ratio:
        size = max(w, h)
        new_image = Image.new("RGB", (size, size), (0, 0, 0))  # Black background
        # paste_x = (size - w) // 2
        # paste_y = (size - h) // 2
        new_image.paste(image, (0, 0))
        return new_image

    # Do nothing
    return image