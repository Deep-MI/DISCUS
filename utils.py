import nibabel as nib 
import numpy as np
import yaml
import torch
import os
from os.path import join
import h5py
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pytorch3d.transforms import quaternion_apply

def read_imagedata(path):

    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header

def read_bvals(path):

    rfile = open(path, "r").readlines()
    if len(rfile) == 1:
        bvals = [x for x in rfile[0].replace("\n", "").split(" ")]
        bvals = np.array([float(x) for x in bvals if len(x) > 0])
    else:
        bvals = np.array([int(x.replace("\n", "")) for x in rfile])
    if len(bvals.shape) == 1:
        bvals = np.expand_dims(bvals, axis=0)
    return bvals

def read_bvecs(path):

    bvecs = open(path, "r").readlines()
    bvecs = np.array([[float(x) for x in bvecs[i].replace("\n", "").replace("\t\t", "\t").replace("\t", " ").split(" ") if len(x) > 0] for i in range(len(bvecs))])
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    return bvecs
    
def write_imagedata(data, path, affine, header=None):

    if header is not None:
        img = nib.Nifti1Image(data, affine, header)
    else:
        img = nib.Nifti1Image(data, affine)
    img.to_filename(path)  


def load_config(filepath):

    with open(filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg

def get_parser():

    parser = ArgumentParser(description = __doc__, formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", dest="filename", help="config file", type=str, required = True)
    return parser

class apply_early_stopping():

    """
    monitor: quantity to be monitored.
    min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
    patience: number of epochs with no improvement after which training will be stopped.
    mode: one of {auto, min}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing
    wait: wait this many epochs to even consider an early stop
    """

    def __init__(self, min_delta=0, patience=0, mode='min', wait=0):
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait = wait
        if mode == "min":
            self.best = np.inf
        elif mode == "max":
            self.best = -np.inf
        self.counter = 0
        self.epochswithoutimprovement = 0
        assert self.mode in ["min", "max"]

    def evaluate(self, monitor):
        self.monitor = monitor
        self.counter += 1

        if self.mode == "max":
            if (self.monitor - self.min_delta) > self.best:
                self.best = self.monitor
                self.epochswithoutimprovement = 0
                return False
            else:
                if self.counter < self.wait:
                    return False
                if self.counter < self.patience:
                    return False
                if self.counter > self.wait and self.counter > self.patience + 1:
                    self.epochswithoutimprovement += 1
                    if self.epochswithoutimprovement >= self.patience:
                        return True
        elif self.mode == "min":
            if (self.monitor + self.min_delta) < self.best:
                self.best = self.monitor
                self.epochswithoutimprovement = 0
                return False
            else:
                if self.counter < self.wait:
                    return False
                if self.counter < self.patience:
                    return False
                if self.counter > self.wait and self.counter > self.patience + 1:
                    self.epochswithoutimprovement += 1
                    if self.epochswithoutimprovement >= self.patience:
                        return True
        return False

def get_dataset(path, test=False):

    file = h5py.File(path, "r")
    if test:
        X1 = np.ones((50000, file.get("bvecs").shape[1], file.get("bvecs").shape[2]))
        X2 = np.ones((50000, file.get("bvals").shape[1], file.get("bvals").shape[2]))
        X3 = np.ones((50000, file.get("signals").shape[1], file.get("signals").shape[2]))
    else:
        X1 = np.array(file.get("bvecs"))
        X2 = np.array(file.get("bvals"))
        X3 = np.array(file.get("signals"))
    file.close()
    return X1, X2, X3

def rotate_batch(x):

    # rotation ~unif(S³) according to Shoemake (1992)

    u,v,w = torch.rand(3)
    sqrtum1 = torch.sqrt(1-u)
    sqrtu = torch.sqrt(u)
    quat = torch.tensor([sqrtum1*torch.sin(2*torch.pi*v),
                     sqrtum1*torch.cos(2*torch.pi*v),
                     sqrtu*torch.sin(2*torch.pi*w),
                     sqrtu*torch.cos(2*torch.pi*w)])

    return torch.transpose(quaternion_apply(quat, torch.transpose(torch.tensor(x),0,1)), 0,1)

def get_best_model_checkpoints(experiment_path):

    print("Searching checkpoint in {} ...".format(experiment_path))
    try:
        checkpoints = [x for x in os.listdir(experiment_path) if ".pkl" in x]
        candidates = np.argwhere(["best" in x for x in checkpoints]).flatten()
        assert len(candidates) == 1
        checkpoint_path = join(experiment_path, checkpoints[candidates[0]])
        print("Choosing checkpoint {} ...".format(checkpoint_path))
    except FileNotFoundError:
        print("No checkpoint ...")
    except AssertionError:
        print("No checkpoint ...")
    return checkpoint_path