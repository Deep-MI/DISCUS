from networks import PredictionWrapper
from utils import load_config, get_best_model_checkpoints, get_parser
from os.path import join, dirname
import sys
import os

def main(args):

    # Load config and set hyperparameters
    cfg = load_config(args.config_file)
    data_path = cfg.get("data_path")
    bvecs_path = cfg.get("bvecs_path")
    bvals_path = cfg.get("bvals_path")
    mask_path = cfg.get("mask_path")
    experiment_path = cfg.get("experiment_path")
    checkpoint_path = get_best_model_checkpoints(experiment_path)
    subject_ids = cfg.get("test_subject_ids")
    device_number = cfg.get("device", 0)
    if torch.cuda.is_available(): device = "cuda:{}".format(device_number)
    else: device = "cpu"
    observation_set_names = cfg.get("observation_set_names")
    observation_set_indices = cfg.get("observation_set_indices")
    validation_batch_size = cfg.get("validation_batch_size", 12000)
    save_path = cfg.get("prediction_path")

    m = PredictionWrapper(checkpoint_path=checkpoint_path,
                          device=device,
                          validation_batch_size=validation_batch_size)

    # Loop over subjects ...
    for subject_id in subject_ids:

        os.makedirs(dirname(save_path).format(subject_id), exist_ok=True)

        m.prepare_data(bvecs_path=bvecs_path.format(subject_id),
                           bvals_path=bvals_path.format(subject_id),
                           data_path=data_path.format(subject_id),
                           mask_path=mask_path.format(subject_id))

        # ... and observation sets
        for observation_set_name, observation_set_index in zip(observation_set_names, observation_set_indices):

            print("Processing subject {} for observation set {} ...".format(subject_id, observation_set_name))
            sys.stdout.flush()

            m.fit_and_predict(observation_set=observation_set_index,
                              query_bvecs_path=None,
                              query_bvals_path=None,
                              save_path=save_path.format(subject_id, observation_set_name))

    with open(join(dirname(dirname(save_path)), "COMPLETED"), "w"): pass
    print("Prediction completed ...")
    sys.stdout.flush()

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)