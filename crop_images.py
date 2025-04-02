import os
import numpy as np
from PIL import Image


base_path = "squares/val/normal" 
output_root = "squares/val/cropped"
classes = ["a", "b", "c"]
target_size = (224, 224)
white_threshold = 235  # Treat pixels with all channels >= 235 as white

def crop_and_resize(image_path):
    try:
        image = Image.open(image_path)
        rgb_array = np.array(image)

        # Identify near-white pixels and treat them as white
        non_white_mask = np.any(rgb_array < white_threshold, axis=-1)
        non_white_coords = np.argwhere(non_white_mask)

        if non_white_coords.size == 0:
            print(f"No colored square found in {image_path}")
            return None

        # Get bounding box of the non-white region
        top_left = non_white_coords.min(axis=0)
        bottom_right = non_white_coords.max(axis=0)

        # Crop the image to the bounding box
        cropped_image = image.crop((top_left[1], top_left[0], bottom_right[1] + 1, bottom_right[0] + 1))

        # Resize to the target size
        resized_image = cropped_image.resize(target_size, Image.LANCZOS)

        return resized_image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


os.makedirs(output_root, exist_ok=True)

#Loop through each class and image, cropping them to 
# colored squares and then saving them as png's
for class_name in classes:
    folder_path = os.path.join(base_path, class_name)
    output_folder = os.path.join(output_root, class_name)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(500):
        for ext in ["bmp", "jpg", "png"]:
            file_path = os.path.join(folder_path, f"{i}.{ext}")
            if os.path.exists(file_path):
                cropped_image = crop_and_resize(file_path)
                if cropped_image:
                    output_path = os.path.join(output_folder, f"{i}.png")
                    cropped_image.save(output_path, format="PNG")
                    print(f"Saved cropped image to {output_path}")
                break
