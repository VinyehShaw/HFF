import os
import numpy as np
import nibabel as nib
import dtcwt
import argparse

def process_nii_lowpass(nii_path, out_path, nlevels=1):
    """
    Perform DTCWT transform on the entire NIfTI file,
    extract only the low-frequency component (lowpass),
    and save it as a new .nii.gz file.
    """
    # Load the NIfTI file
    nii = nib.load(nii_path)
    data = nii.get_fdata()  # Assume data is 3D with shape (height, width, slices)
    print(f'  Original data shape: {data.shape}')
    
    # Initialize the DTCWT transformer
    transform = dtcwt.Transform2d()
    
    lowpass_slices = []
    # Apply DTCWT to each slice and extract the low-frequency component
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        transformed = transform.forward(slice_img, nlevels=nlevels)
        LL = transformed.lowpass  # Low-frequency component
        # Normalize to [0, 255]
        LL_norm = (LL - LL.min()) / (LL.max() - LL.min()) * 255
        lowpass_slices.append(LL_norm.astype(np.uint8))
    
    # Stack all slices back into a 3D array
    lowpass_data = np.stack(lowpass_slices, axis=2)
    print(f'  Low-frequency data shape: {lowpass_data.shape}')
    
    # Create a new NIfTI image and save it
    new_nii = nib.Nifti1Image(lowpass_data, affine=nii.affine)
    nib.save(new_nii, out_path)
    print(f'  Low-frequency result saved to: {out_path}')

def main(data_root):
    # Iterate over all patient folders in the data root directory
    for patient in os.listdir(data_root):
        patient_path = os.path.join(data_root, patient)
        if not os.path.isdir(patient_path):
            continue
        print(f"Processing patient folder: {patient_path}")
        
        # Iterate over all .nii and .nii.gz files in the patient folder
        for file in os.listdir(patient_path):
            # Only process .nii or .nii.gz files
            if not (file.endswith('.nii') or file.endswith('.nii.gz')):
                continue
            # Skip segmentation files or already processed results
            if 'seg' in file.lower() or '_h' in file.lower() or '_l' in file.lower():
                continue
            
            nii_path = os.path.join(patient_path, file)
            print(f"  Processing modality file: {file}")
            
            # Determine base filename without extension
            if file.endswith('.nii.gz'):
                base = file[:-7]  # Remove ".nii.gz"
            elif file.endswith('.nii'):
                base = file[:-4]  # Remove ".nii"
            else:
                base = file
            
            out_file = base + '_L.nii.gz'
            out_path = os.path.join(patient_path, out_file)
            
            process_nii_lowpass(nii_path, out_path, nlevels=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract low-frequency features from NIfTI files using DTCWT"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/mcga/phd/brats2020/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/',
        help='Path to the root directory containing patient data'
    )
    args = parser.parse_args()
    main(args.data_root)
