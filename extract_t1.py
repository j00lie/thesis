import os
import shutil


def extract_t1_images_and_masks(source_folder, target_folder):
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all subfolders in the source directory
    subfolders = [
        d
        for d in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, d))
    ]

    for subfolder in subfolders:
        source_subfolder_path = os.path.join(source_folder, subfolder)

        t1_image_path = os.path.join(source_subfolder_path, "vs_gk_t1_refT1.nii.gz")
        t1_contour_path = os.path.join(source_subfolder_path, "vs_gk_seg_refT1.nii.gz")

        if os.path.exists(t1_image_path) and os.path.exists(t1_contour_path):
            # Create a new subfolder in target directory
            target_subfolder_path = os.path.join(target_folder, subfolder)
            if not os.path.exists(target_subfolder_path):
                os.makedirs(target_subfolder_path)

            # Copy T1 image and mask to the new subfolder
            shutil.copy(t1_image_path, target_subfolder_path)
            shutil.copy(t1_contour_path, target_subfolder_path)


if __name__ == "__main__":
    source_folder_path = (
        r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti"
    )
    target_folder_path = (
        r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti_t1"
    )
    extract_t1_images_and_masks(source_folder_path, target_folder_path)
