import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import segmentation_models_3D as sm

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

    return resized_data, data.shape


mri_directory_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti_t1\vs_gk_*[0-9]*\vs_gk_t1_refT1.nii.gz"
mri_paths = sorted(glob(mri_directory_path, recursive=True))

mask_directory_path = r"C:\Users\joona\Documents\LUT\bsc_thesis\bsc_thesis_data_nifti_t1\vs_gk_*[0-9]*\vs_gk_seg_refT1.nii.gz"
mask_paths = sorted(glob(mask_directory_path, recursive=True))


def has_duplicates(lst):
    return len(lst) != len(set(lst))


print("The MRI paths contain duplicates: " + str(has_duplicates(mri_paths)))
print("The Mask paths contain duplicates: " + str(has_duplicates(mask_paths)))

mris = []
mri_dims = []
for path in tqdm(mri_paths):
    mri, orig_dims = load_and_resize_nifti(path)
    mris.append(mri)
    mri_dims.append(orig_dims)
masks = []
mask_dims = []
for path in tqdm(mask_paths):
    mask, orig_mask_dims = load_and_resize_nifti(path)
    masks.append(mask)
    mask_dims.append(orig_mask_dims)


# Convert list of tuples to numpy array
mri_dims_arr = np.array(mri_dims)
mask_dims_arr = np.array(mask_dims)


def print_shapes(dim_array):
    # Extract unique widths and their counts
    unique_widths, counts_widths = np.unique(dim_array[:, 0], return_counts=True)

    # Extract unique heights and their counts
    unique_heights, counts_heights = np.unique(dim_array[:, 1], return_counts=True)

    # Extract unique depths and their counts
    unique_depths, counts_depths = np.unique(dim_array[:, 2], return_counts=True)

    print("Unique Widths and Their Counts:")
    for width, count in zip(unique_widths, counts_widths):
        print(f"{width}: {count}")

    print("\nUnique Heights and Their Counts:")
    for height, count in zip(unique_heights, counts_heights):
        print(f"{height}: {count}")

    print("\nUnique Depths and Their Counts:")
    for depth, count in zip(unique_depths, counts_depths):
        print(f"{depth}: {count}")


print_shapes(mri_dims_arr)
print_shapes(mask_dims_arr)

for i, mask in enumerate(masks):
    masks[i] = np.array(mask, dtype=bool)


def imgSanity(imgArray, maskArray):
    # Randomly select an image stack
    r_numb = random.randrange(0, len(imgArray))

    img_stack = imgArray[r_numb]
    mask_stack = maskArray[r_numb]

    # Loop until a slice with a mask is found
    while True:
        r_numb_2 = random.randrange(0, IMG_DEPTH)  # choose random slice from stack
        mask_slice = mask_stack[:, :, r_numb_2]

        # Check if the mask slice contains any True values (indicating a mask)
        if np.any(mask_slice):
            break

    # Extract the chosen slice from the image
    img_slice = img_stack[:, :, r_numb_2]

    # Display the image slice
    plt.imshow(img_slice, cmap="gray")

    # # Create a colored overlay using the jet colormap only for the mask region
    # overlay = plt.cm.jet(mask_slice.astype(float))

    # # Set the alpha channel to zero where the mask is False (making it transparent)
    # overlay[~mask_slice, -1] = 0  # Set alpha channel to zero

    # # Overlay the colored mask on the image
    # plt.imshow(overlay, alpha=0.5)
    # Overlay the contour on the image
    plt.contour(mask_slice, colors="r", linewidths=0.5)
    plt.show()


imgSanity(mris, masks)


def standardize_3d_image(img):
    """
    Standardize a 3D image using Z-score normalization.

    Parameters:
    - img: 3D numpy array representing the MRI image.

    Returns:
    - standardized_img: 3D numpy array representing the standardized MRI image.
    """
    mean = np.mean(img)
    std = np.std(img)
    standardized_img = (img - mean) / std
    return standardized_img


for idx, mri in enumerate(mris):
    mris[idx] = standardize_3d_image(mri)


X_train, X_test, Y_train, Y_test = train_test_split(
    mris, masks, test_size=0.20, shuffle=True
)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)

print(
    f"Training data\nDimensions: {X_train.shape}\nMax-value: {np.max(X_train)}\nMin-value: {np.min(X_train)}"
)
print(
    f"Test data\nDimensions: {X_test.shape}\nMax-value: {np.max(X_test)}\nMin-value: {np.min(X_test)}"
)


model = sm.Unet(
    "resnet34",
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH),
    encoder_weights=None,
    activation="sigmoid",
)


dice_loss = sm.losses.DiceLoss()
bce_loss = sm.losses.BinaryCELoss()
total_loss = dice_loss + bce_loss


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=total_loss,
    metrics=[sm.metrics.IOUScore()],
)


cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_iou_score", mode="max", patience=10, restore_best_weights=True
)


history = model.fit(
    X_train, Y_train, epochs=100, batch_size=1, validation_split=0.20, callbacks=[cb]
)
