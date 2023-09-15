"""
Function for automatically merging the contours 
and the registration matrices with the image data
"""

import os
import shutil


def move_contours_to_data(data_folder, contours_folder):
    # List all subfolders in the contours directory
    contour_subfolders = [
        d
        for d in os.listdir(contours_folder)
        if os.path.isdir(os.path.join(contours_folder, d))
    ]

    for subfolder in contour_subfolders:
        source_folder = os.path.join(contours_folder, subfolder)
        target_folder = os.path.join(data_folder, subfolder)

        if os.path.exists(target_folder):
            for contour_file in os.listdir(source_folder):
                source_file = os.path.join(source_folder, contour_file)
                target_file = os.path.join(target_folder, contour_file)

                shutil.move(source_file, target_file)
        else:
            print(f"No matching folder for {subfolder} in the data directory!")


if __name__ == "__main__":
    data_folder_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data"
    contours_folder_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\Vestibular-Schwannoma-SEG_matrices Mar 2021\registration_matrices"  # r"C:\Users\joona\Documents\LUT\bsc_thesis\Vestibular-Schwannoma-SEG contours Mar 2021\contours"
    move_contours_to_data(data_folder_path, contours_folder_path)
