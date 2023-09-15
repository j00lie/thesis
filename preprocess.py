import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 64


def load_and_resize_nifti(file_path, target_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)):
    # Load the image with memmap option for efficient reading
    nifti_img = nib.load(file_path, mmap=True)

    # Get the data as a memory-mapped array
    data = nifti_img.get_fdata()

    # Calculate the zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]

    # Resize the image
    resized_data = zoom(
        data, zoom_factors, order=1
    )  # Use bilinear interpolation (order=1)

    return resized_data


mri_directory_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti_t1\vs_gk_*[0-9]*\vs_gk_t1_refT1.nii.gz"
mri_paths = sorted(glob(mri_directory_path, recursive=True))

mask_directory_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti_t1\vs_gk_*[0-9]*\vs_gk_seg_refT1.nii.gz"
mask_paths = sorted(glob(mask_directory_path, recursive=True))


def has_duplicates(lst):
    return len(lst) != len(set(lst))


print(has_duplicates(mri_paths))

mris = []
for path in tqdm(mri_paths):
    mris.append(load_and_resize_nifti(path))


def imgSanity(imgArray):
    # random pet-mri-mask generator
    r_numb = random.randrange(0, len(imgArray))  # choose random stack
    r_numb_2 = random.randrange(0, IMG_DEPTH)  # choose random slice from stack
    plt.imshow(imgArray[r_numb][:, :, r_numb_2])
