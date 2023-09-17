# Official Github repository for the B.Sc. thesis by Joonas Liedes titled: "Automatic segmentation of vestibular schwannoma using deep learning" 


## Data
- Download the [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053) using the [NBIA data retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-DownloadingtheNBIADataRetriever)

## Pre-processing
- Organise the folder structure using this [script](https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing#create-data-set-with-convenient-folder-structure)
- Merge contours and transformation matrices to their respective image folders by running [merge_contours.py](https://github.com/j00lie/thesis/blob/main/merge_contours.py)
- Convert files to nifti using this [script](https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing#conversion-of-dicom-images-and-contoursjson-files-to-nifti-and-optional-registration)
- Extract the T1 MRI sequences by running the [extract_t1.py](https://github.com/j00lie/thesis/blob/main/extract_t1.py)
- Resize and save images as numpy arrays by following along the [preprocessing notebook](https://github.com/j00lie/thesis/blob/main/preprocess.ipynb)

## Training and Evaluating the model
- Follow along the [training notebook](https://github.com/j00lie/thesis/blob/main/train.ipynb)
  - Note: For running in Google Colab connect your github to your Colab and upload the preprocessed numpy files to your drive
- 5-fold cross-validation implemented [here](https://github.com/j00lie/thesis/blob/main/train_5foldCV.ipynb) 
