import os
import shutil
from glob import glob

def flatten_vos_split(input_dir, output_dir):
    img_root = os.path.join(input_dir, 'JPEGImages')
    mask_root = os.path.join(input_dir, 'Annotations')

    out_img_dir = os.path.join(output_dir, 'images')
    out_mask_dir = os.path.join(output_dir, 'masks')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Flatten picture
    img_paths = glob(os.path.join(img_root, '*', '*.jpg'))
    for path in img_paths:
        video_id = os.path.basename(os.path.dirname(path))
        frame = os.path.basename(path)
        new_name = f"{video_id}_{frame}"
        shutil.copy(path, os.path.join(out_img_dir, new_name))

    # Flatten mask
    mask_paths = glob(os.path.join(mask_root, '*', '*.png'))
    for path in mask_paths:
        video_id = os.path.basename(os.path.dirname(path))
        frame = os.path.basename(path)
        new_name = f"{video_id}_{frame}"
        shutil.copy(path, os.path.join(out_mask_dir, new_name))

    print(f"Flattened {len(img_paths)} images and {len(mask_paths)} masks into {output_dir}/")

# Main function to run the script
if __name__ == "__main__":
    for part in ['train','test','valid']:
        flatten_vos_split(
            input_dir=f"./data/youtube_vos/{part}", 
            output_dir=f"./data/youtube_vos_flat/{part}"
        )
