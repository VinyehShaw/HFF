import os
import random
import numpy as np
from utils import check_dir

def correct_path(path):
    return path.replace('\\', '/').replace('../dataset', './dataset')

def split(brain_dir, save_dir, n_split=3):
    
    # Specify the folder to be segmented as the sample folder under brain_dir
    sample_dir = os.path.join(brain_dir)
    patient_ids = os.listdir(sample_dir)  # Get all files/folders under the sample folder
    np.random.seed(9418)  #   Set random seed

    check_dir(save_dir)
    
    for split_index in range(n_split):
        #  Randomly shuffle the order of patient IDs
        np.random.shuffle(patient_ids)
        
        total = len(patient_ids)
        n_train = int(total * 0.9)
        # n_val = total - n_train
        n_val = int(total * 0.95)
        # Split the patient IDs into training and validation sets
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[int(total * 0.9):]

        # Create the train and validation file paths
        train_file_path = os.path.join(save_dir, f'{split_index}-train.txt')
        val_file_path = os.path.join(save_dir, f'{split_index}-val.txt')

        with open(train_file_path, 'w') as train_f:
            for pid in train_ids:
                path = os.path.join(sample_dir, pid)
                corrected_path = correct_path(path)
                train_f.write(f"{corrected_path}\n")

        with open(val_file_path, 'w') as val_f:
            for pid in val_ids:
                path = os.path.join(sample_dir, pid)
                corrected_path = correct_path(path)
                val_f.write(f"{corrected_path}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain_dir', type=str, default='/home/mcga/phd/brain/')
    parser.add_argument('--save_dir', type=str, default='./brats19')
    parser.add_argument('--n_split', type=int, default=3)
    parser.add_argument('--seed', type=int, default=9418)
    options = parser.parse_args()

    random.seed(options.seed)
    np.random.seed(options.seed)

    split(options.brain_dir, options.save_dir, n_split=options.n_split)
    print('Done!')
