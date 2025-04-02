import os
import numpy as np
from PIL import Image

base_path = "squares/train/normal"  
classes = ["a", "b", "c"]

def get_square_size(image_path):
    try:
        image = Image.open(image_path)
        rgb_array = np.array(image)

        # Identify non-white pixels (where at least one channel is not 255)
        non_white_mask = np.any(rgb_array != 255, axis=-1)
        non_white_coords = np.argwhere(non_white_mask)

        # Get bounding box of the non-white region, get width and height
        top_left = non_white_coords.min(axis=0)
        bottom_right = non_white_coords.max(axis=0)
        width = bottom_right[1] - top_left[1] + 1
        height = bottom_right[0] - top_left[0] + 1

        # Find % of image, square takes up
        square_size = (width + height) / 2
        image_size = rgb_array.shape[0] * rgb_array.shape[1]
        square_percentage = (square_size ** 2) / image_size * 100

        return square_size, square_percentage
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

for class_name in classes:
    sizes = []
    percentages = []
    folder_path = os.path.join(base_path, class_name)
    for i in range(500):
        # Find correct file type
        for ext in ["bmp", "jpg", "png"]:
            file_path = os.path.join(folder_path, f"{i}.{ext}")
            if os.path.exists(file_path):
                size, percentage = get_square_size(file_path)
                if size is not None:
                    sizes.append(size)
                    percentages.append(percentage)
                break

    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    mean_percentage = np.mean(percentages)

    print(f"Class {class_name}:")
    print(f"  Mean square size = {mean_size:.2f}")
    print(f"  Std dev = {std_size:.2f}")
    print(f"  Min square size = {min_size:.2f}")
    print(f"  Max square size = {max_size:.2f}")
    print(f"  Mean % of image occupied = {mean_percentage:.2f}%\n")
