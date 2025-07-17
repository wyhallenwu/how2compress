# from PIL import Image
# import numpy as np
# import cv2


# def apply_perspective_transform(image, src_points, dst_points):
#     """
#     Applies a perspective transform to an image.

#     :param image: The input PIL image.
#     :param src_points: A list of four points (tuples) in the original image.
#     :param dst_points: A list of four points (tuples) in the output image.
#     :return: The transformed PIL image.
#     """
#     # Convert the points to numpy arrays
#     src_array = np.array(src_points, dtype=np.float32)
#     dst_array = np.array(dst_points, dtype=np.float32)

#     # Calculate the perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(src_array, dst_array)

#     # Apply the perspective transform
#     transformed_image = cv2.warpPerspective(
#         np.array(image),
#         matrix,
#         (image.width, image.height),
#     )

#     return Image.fromarray(transformed_image)


# # Load the image
# image_path = "/how2compress/data/MOT17Det/train/MOT17-04/img1/001040.jpg"
# image = Image.open(image_path)

# # Define source points from the original image (corners of the image)
# src_points = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]

# # Define destination points for the perspective transformation
# # These points will create the skewed effect.
# dst_points = [
#     (0, image.height // 4),
#     (image.width * 0.25, 0),
#     (image.width * 0.25, image.height // 4 * 3),
#     (0, image.height),
# ]

# # Apply perspective transformation
# transformed_image = apply_perspective_transform(image, src_points, dst_points)

# # Save or display the image
# # transformed_image.show()
# transformed_image.save("test.png")


from PIL import Image
import numpy as np
import cv2


def apply_perspective_transform(image, src_points, dst_points):
    """
    Applies a perspective transform to an image and preserves transparency.
    :param image: The input PIL image.
    :param src_points: A list of four points (tuples) in the original image.
    :param dst_points: A list of four points (tuples) in the output image.
    :return: The transformed PIL image with transparency.
    """
    # Convert the image to RGBA if it's not already
    image = image.convert("RGBA")

    # Convert the points to numpy arrays
    src_array = np.array(src_points, dtype=np.float32)
    dst_array = np.array(dst_points, dtype=np.float32)

    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_array, dst_array)

    # Split the image into color channels and alpha channel
    color_channels = np.array(image.convert("RGB"))
    alpha_channel = np.array(image.split()[3])

    # Apply the perspective transform to color channels
    transformed_color = cv2.warpPerspective(
        color_channels,
        matrix,
        (image.width, image.height),
    )

    # Apply the perspective transform to alpha channel
    transformed_alpha = cv2.warpPerspective(
        alpha_channel,
        matrix,
        (image.width, image.height),
    )

    # Merge the transformed color channels with the transformed alpha channel
    transformed_image = np.dstack((transformed_color, transformed_alpha))

    return Image.fromarray(transformed_image)


# Load the image
image_path = "g2.png"
image = Image.open(image_path).convert("RGBA")

# Define source points from the original image (corners of the image)
src_points = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]

# Define destination points for the perspective transformation
# These points will create the skewed effect.
dst_points = [
    (0, image.height // 4),
    (image.width * 0.25, 0),
    (image.width * 0.25, image.height // 4 * 3),
    (0, image.height),
]

# Apply perspective transformation
transformed_image = apply_perspective_transform(image, src_points, dst_points)

# Save the image with transparency
transformed_image.save("g2-trans.png")
