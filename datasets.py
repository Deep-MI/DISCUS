from utils import read_imagedata, read_bvals, read_bvecs, get_parser, load_config
import numpy as np
import h5py
import os

def main(subject_ids):

    X1 = []
    X2 = []
    X3 = []
     
    for subject_id in subject_ids:

        print("Processing subject {} ...".format(subject_id))
        
        # Load normalized data, b-values and b-vectors, assume that DWIs are stacked along the fourth axis
        data,_,_ = read_imagedata(normalized_data_path.format(subject_id))
        assert len(data.shape) == 4
        data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2], 1, data.shape[3])
        bvals = read_bvals(bvals_path.format(subject_id))/1000
        bvecs = read_bvecs(bvecs_path.format(subject_id))
        assert data.shape[-1] == bvals.shape[-1] == bvecs.shape[-1]

        # Load white matter / grey matter mask    
        mask,_,_ = read_imagedata(mask_path.format(subject_id))
        mask = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2])

        # Get signals corresponding to white matter / grey matter voxels
        data = data[mask == 1, :, :]

        X1.append(np.stack(data.shape[0] * [bvecs], axis=0))
        X2.append(np.stack(data.shape[0] * [bvals], axis=0))
        X3.append(data)
    
    # Convert lists to numpy arrays
    X1 = np.concatenate(X1, axis=0)
    X2 = np.concatenate(X2, axis=0)
    X3 = np.concatenate(X3, axis=0)
    return X1, X2, X3


if __name__ == '__main__':

    args = get_parser().parse_args()
    cfg = load_config(args.config_file)
    bvals_path = cfg.get("bvals_path")
    bvecs_path = cfg.get("bvecs_path")
    mask_path = cfg.get("mask_path")
    normalized_data_path = cfg.get("data_path")
    training_subjects = cfg.get("training_subject_ids")
    validation_subjects = cfg.get("validation_subject_ids")
    training_dataset_save_path = cfg.get("training_dataset_path")
    validation_dataset_path = cfg.get("validation_dataset_path")

    print("Creating training and validation sets ...")
    os.makedirs(os.path.dirname(training_dataset_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(validation_dataset_path), exist_ok=True)

    bvecs, bvals, signals = main(subject_ids = training_subjects)
    print("Training Set bvecs.shape: ", bvecs.shape)
    print("Training Set bvals.shape: ", bvals.shape)
    print("Training Set signals.shape: ", signals.shape)
    hf = h5py.File(training_dataset_save_path, "w")
    hf.create_dataset('bvecs', data=bvecs, compression='gzip')
    hf.create_dataset('bvals', data=bvals, compression='gzip')
    hf.create_dataset('signals', data=signals, compression='gzip')
    hf.close()

    bvecs, bvals, signals = main(subject_ids = validation_subjects)
    print("Validation Set bvecs.shape: ", bvecs.shape)
    print("Validation Set bvals.shape: ", bvals.shape)
    print("Validation Set signals.shape: ", signals.shape)
    hf = h5py.File(validation_dataset_path, "w")
    hf.create_dataset('bvecs', data=bvecs, compression='gzip')
    hf.create_dataset('bvals', data=bvals, compression='gzip')
    hf.create_dataset('signals', data=signals, compression='gzip')
    hf.close()

    print("Successfully created the training and validation sets ...")
