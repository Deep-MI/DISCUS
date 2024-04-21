import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks import DISCUS, TrainingDataloader, ValidationDataloader
import json
import os
from os.path import join
import sys
import numpy as np
from utils import load_config, get_parser, apply_early_stopping,get_dataset

def main(yaml_filepath):

    # Load config file and set hyperparameters
    cfg = load_config(yaml_filepath)
    query_fraction = cfg.get("query_fraction", 0.4)

    device_number = cfg.get("device", 0)
    if torch.cuda.is_available(): device = "cuda:{}".format(device_number)
    else: device = "cpu"

    max_epochs = cfg.get("max_epochs", 100)
    experiment_path = cfg.get("experiment_path")
    training_batch_size = cfg.get("training_batch_size", 4000)
    validation_batch_size = cfg.get("validation_batch_size", 12000)

    # Observation set for the validation set
    validation_indices = cfg.get("validation_indices")
    os.makedirs(experiment_path, exist_ok=True)
    init_lr = cfg.get("lr", 0.01)

    # Declare the model
    model = DISCUS().to(device)

    # Declare optimizer, early stopping, learning rate scheduler, loss, augmentations
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    es = apply_early_stopping(min_delta=0.00001, patience=20, mode='min', wait=20)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=5e-5)

    # Load and check data
    bvecs, bvals, signals = get_dataset(path=cfg["training_dataset_path"], test=False)
    dataset_train = TrainingDataloader(bvecs=bvecs, bvals=bvals, signals=signals)
    train_dataloader = DataLoader(dataset=dataset_train, batch_size=training_batch_size, shuffle=True)

    bvecs, bvals, signals = get_dataset(path=cfg["validation_dataset_path"], test=False)
    dataset_validation = ValidationDataloader(bvecs=bvecs, bvals=bvals, signals=signals, validation_indices=validation_indices)
    validation_dataloader = DataLoader(dataset=dataset_validation, batch_size=validation_batch_size, shuffle=False)

    training_dict = dict()
    training_dict["epoch"] = []
    training_dict["lr"] = []
    training_dict["training_loss"] = []
    training_dict["validation_loss"] = []
    sys.stdout.flush()

    for epoch in range(max_epochs):
        
        print("Epoch: {}, LR {}".format(epoch, optimizer.param_groups[0]["lr"]))
        
        model.train()
        train_loss = 0
        batch_counter = 0

        for batch in train_dataloader:

            optimizer.zero_grad()
            _, prediction_signals = model.forward(batch["observation_bvecs"].float().to(device),
                                                  batch["observation_bvals"].float().to(device),
                                                  batch["observation_signals"].float().to(device),
                                                  batch["observation_mask"].to(device),
                                                  batch["query_bvecs"].float().to(device),
                                                  batch["query_bvals"].float().to(device))

            loss = (prediction_signals - batch["reference_signals"].float().to(device))**2
            observation_loss = loss[batch["reference_mask"].to(device)].mean()
            query_loss = loss[~batch["reference_mask"].to(device)].mean()
            weighted_loss = query_fraction * query_loss + (1 - query_fraction) * observation_loss
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.detach().cpu().numpy()
            batch_counter += 1

        train_loss /= batch_counter
        print("Epoch {}: training-loss {}".format(epoch, train_loss))
        sys.stdout.flush()

        with torch.no_grad():

            model.eval()
            validation_loss = 0
            batch_counter = 0

            for batch in validation_dataloader:

                _, prediction_signals = model.forward(batch["observation_bvecs"].float().to(device),
                                                      batch["observation_bvals"].float().to(device),
                                                      batch["observation_signals"].float().to(device),
                                                      batch["observation_mask"].to(device),
                                                      batch["query_bvecs"].float().to(device),
                                                      batch["query_bvals"].float().to(device))

                loss = (prediction_signals - batch["reference_signals"].float().to(device)) ** 2
                observation_loss = loss[batch["reference_mask"].to(device)].mean()
                query_loss = loss[~batch["reference_mask"].to(device)].mean()
                weighted_loss = query_fraction * query_loss + (1 - query_fraction) * observation_loss
                validation_loss += weighted_loss.detach().cpu().numpy()
                batch_counter += 1

            validation_loss /= batch_counter

        # Check if this is a new best model checkpoint
        save_name = join(experiment_path, "epoch_{}_best.pkl".format(str(epoch).zfill(2)))
        message = "Best Epoch {}: validation-loss {}".format(epoch, validation_loss)

        if len(training_dict["validation_loss"]) > 0:
            if validation_loss < np.min(training_dict["validation_loss"]):
                existing_checkpoints = [join(experiment_path, x) for x in os.listdir(experiment_path) if x[-4:] == ".pkl" and "_best" in x]
                for file in existing_checkpoints: os.rename(file, file.replace("_best", ""))
            else:
                save_name = join(experiment_path, 'epoch_{}.pkl'.format(str(epoch).zfill(2))) 
                message = "Epoch {}: validation-loss {}".format(epoch, validation_loss)
        print(message)

        sys.stdout.flush()

        # Save model checkpoint
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "epoch": epoch}

        torch.save(checkpoint, save_name)
        training_dict["epoch"].append(epoch)
        training_dict["lr"].append(optimizer.param_groups[0]["lr"])
        training_dict["training_loss"].append(train_loss.item())
        training_dict["validation_loss"].append(validation_loss.item())

        with open(join(experiment_path, "validation_dict.json"), 'w') as fp: json.dump(training_dict, fp)

        # Update learning rate with the scheduler
        scheduler.step(validation_loss)

        # Check if early stopping is triggered
        if es.evaluate(validation_loss):
            print("Applying early stopping ...")
            file = open(join(experiment_path, "COMPLETED"), "w")
            file.close()
            break

if __name__ == '__main__':

    args = get_parser().parse_args()
    config_file = args.filename
    main(config_file)