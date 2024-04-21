import torch
import numpy as np
from torch.nn import Tanh, ReLU
from torch.utils.data.dataloader import DataLoader
from utils import read_imagedata, read_bvecs, read_bvals, write_imagedata, rotate_batch
from torch import nn

class PredictionWrapper:

    def __init__(self, device, checkpoint_path, validation_batch_size):
        self.model = DISCUS()
        self.device = device
        self.model = self.model.to(self.device)
        self.batch_size = validation_batch_size
        checkpoint = torch.load(checkpoint_path, map_location='{}'.format(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

    def prepare_data(self, bvecs_path, bvals_path, data_path, mask_path=None):

        # Load entire data beforehand s.t. multiple observation sets can be built from the data
        data, self.affine, self.header = read_imagedata(data_path)
        self.data_shape = data.shape
        bvals = read_bvals(bvals_path) / 1000
        bvecs = read_bvecs(bvecs_path)

        if mask_path is None:
            mask_init = np.ones((data.shape[0], data.shape[1], data.shape[2]))
        else:
            mask_init, _, _ = read_imagedata(mask_path)

        self.mask = mask_init.reshape(mask_init.shape[0] * mask_init.shape[1] * mask_init.shape[2])
        data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2], 1, data.shape[3])
        self.data = data[self.mask == 1, :, :]
        self.bvecs = np.stack(data.shape[0]*[bvecs], axis=0)
        self.bvals = np.stack(data.shape[0]*[bvals], axis=0)

    def fit_and_predict(self, observation_set, query_bvecs_path, query_bvals_path, save_path):

        dataset_test = ValidationDataloader(bvecs=self.bvecs, bvals=self.bvals, signals=self.data, validation_indices=observation_set,
                                            query_bvecs=query_bvecs_path, query_bvals=query_bvals_path)

        test_dataloader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False)

        prediction = []
        with torch.no_grad():
            for batch in enumerate(test_dataloader):

                _, prediction_signals = self.model.forward(batch["observation_bvecs"].float().to(self.device),
                                                      batch["observation_bvals"].float().to(self.device),
                                                      batch["observation_signals"].float().to(self.device),
                                                      batch["observation_mask"].to(self.device),
                                                      batch["query_bvecs"].float().to(self.device),
                                                      batch["query_bvals"].float().to(self.device))
                prediction.append(prediction_signals.detach().cpu().numpy())

            prediction = np.concatenate(prediction, axis=0)

        if len(prediction.shape) == 2: prediction = np.expand_dims(prediction, axis=1)
        pred = np.zeros((self.mask.shape[0], prediction.shape[1], prediction.shape[2]))
        pred[self.mask == 1, ...] = prediction
        prediction = pred.reshape(self.data_shape[0], self.data_shape[1], self.data_shape[2], prediction.shape[2])

        # Don't save the  prediction for the origin
        prediction = prediction[...,:-1]
        prediction_block = prediction
        write_imagedata(prediction_block, save_path, self.affine, self.header)


class TrainingDataloader():

    def __init__(self, bvecs, bvals, signals):
        self.bvecs = bvecs
        self.bvals = bvals
        self.signals = signals
        self.count = self.bvecs.shape[0]
        
    def __getitem__(self, index):

        # rotate b-vectors
        bvecs = self.bvecs[index]
        rot_bvecs = rotate_batch(bvecs)
        bvals = self.bvals[index]

        # create a mask for the random observation set
        thresholds = ((torch.rand(1) * (bvals.shape[-1] - 6) + 6) / bvals.shape[-1])
        mask = torch.rand(bvals.shape)
        mask = mask < thresholds

        # invert mask from indicating observation vectors to indicating their complement, i.e. vector for which signals need to be mapped out in the network
        observation_mask = ~mask
        observation_signals = self.signals[index]

        # reference signals = signals with ones (target for the q-space origin)
        reference_signals = np.concatenate([observation_signals, np.ones((1,1))], axis=1)

        # reference mask (see above)
        reference_mask = np.concatenate([mask,  np.array([[False]])], axis=1)

        # query all bvecs and bvals and append origin point (bvec = (0,0,1), bval=0)
        query_bvecs = np.concatenate([rot_bvecs, np.stack(1*[np.array([0., 0., 1.])], axis=1)],axis=1)
        query_bvals = np.concatenate([bvals, np.stack(1*[np.array([0.])], axis=1)], axis=1)

        d = dict()
        d["observation_bvecs"] = rot_bvecs
        d["observation_bvals"] = bvals
        d["observation_signals"] = observation_signals
        d["query_bvecs"] = query_bvecs
        d["query_bvals"] = query_bvals
        d["reference_signals"] = reference_signals
        d["observation_mask"] = observation_mask
        d["reference_mask"] = reference_mask
        return d
    
    def __len__(self):
        return self.count


class ValidationDataloader():

    def __init__(self, bvecs, bvals, signals, validation_indices, query_bvecs=None, query_bvals=None):

        self.bvecs = bvecs
        self.bvals = bvals
        if query_bvals is None and query_bvecs is None:
            self.query_bvecs = bvecs
            self.query_bvals = bvals
            self.return_loss_objects = True
        else:
            assert query_bvals is not None and query_bvecs is not None
            self.query_bvecs = query_bvecs
            self.query_bvals = query_bvals
            self.return_loss_objects = False
        self.signals = signals
        self.validation_indices = validation_indices
        self.count = self.bvecs.shape[0]

    def __getitem__(self, index):

        observation_bvecs = self.bvecs[index]
        observation_bvals = self.bvals[index]

        # create a mask for the fixed observation set
        mask = np.full((1, len(self.validation_indices)), True)

        # invert mask from indicating observation vectors to indicating their complement, i.e. vector for which signals need to be mapped out in the network
        observation_mask = ~mask

        observation_signals = self.signals[index][:, self.validation_indices]
        observation_bvecs = observation_bvecs[:, self.validation_indices]
        observation_bvals = observation_bvals[:, self.validation_indices]

        # query all bvecs and bvals and append origin point (bvec = (0,0,1), bval=0)
        query_bvecs = np.concatenate([self.query_bvecs[index], np.stack(1 * [np.array([0., 0., 1.])], axis=1)], axis=1)
        query_bvals = np.concatenate([self.query_bvals[index], np.stack(1 * [np.array([0.])], axis=1)], axis=1)

        if self.return_loss_objects:

            # reference signals = signals with ones (target for the q-space origin)
            reference_signals = np.concatenate([self.signals[index], np.ones((1, 1))], axis=1)
            # reference mask (see above)
            # create a mask for the fixed observation set
            reference_mask = np.full(self.query_bvals[index].shape, False)
            reference_mask[..., self.validation_indices] = True
            reference_mask = np.concatenate([reference_mask, np.array([[False]])], axis=1)

        d = dict()
        d["observation_bvecs"] = observation_bvecs
        d["observation_bvals"] = observation_bvals
        d["observation_signals"] = observation_signals
        d["query_bvecs"] = query_bvecs
        d["query_bvals"] = query_bvals
        d["observation_mask"] = observation_mask
        if self.return_loss_objects:
            d["reference_signals"] = reference_signals
            d["reference_mask"] = reference_mask
        else:
            d["reference_signals"] = None
            d["reference_mask"] = None
        return d

    def __len__(self):
        return self.count


class DISCUS(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_activation = Tanh()
        self.decoder_activation = ReLU()
        self.encoder_batchnorm = True
        self.decoder_batchnorm = False
        si_dims = [3,128, 128, 128, 128]
        encoder_dims = [[130,128,128,128],[256,128,128,128],[256,128,128,16]]
        decoder_dims = [145,128,128,1]

        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks_bn = nn.ModuleList()
        for block_id in range(len(encoder_dims)):
            block = nn.ModuleList()
            block_bn = nn.ModuleList()
            for layer_id in range(len(encoder_dims[block_id])-1):
                block.append(nn.Conv1d(encoder_dims[block_id][layer_id], encoder_dims[block_id][layer_id+1], 1, padding=0))
                block_bn.append(nn.BatchNorm1d(encoder_dims[block_id][layer_id+1]))
            self.encoder_blocks.append(block)
            self.encoder_blocks_bn.append(block_bn)

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers_bn = nn.ModuleList()
        for layer_id in range(len(decoder_dims)-1):
            self.decoder_layers.append(nn.Conv1d(decoder_dims[layer_id], decoder_dims[layer_id+1], 1, padding=0))
            self.decoder_layers_bn.append(nn.BatchNorm1d(decoder_dims[layer_id+1]))

        # Layers of the sign invariant module
        self.si_l1 = nn.Conv1d(si_dims[0], si_dims[1], 1, padding=0)
        self.si_l2 = nn.Conv1d(si_dims[1], si_dims[2], 1, padding=0)
        self.si_l3 = nn.Conv1d(si_dims[2], si_dims[3], 1, padding=0)
        self.si_l4 = nn.Conv1d(si_dims[3], si_dims[4], 1, padding=0)
        self.si_activation = ReLU()

    def forward(self, observation_bvecs, observation_bvals, observation_signals, mask, query_bvecs, query_bvals):

        observation_bvecs_si = self.si(observation_bvecs)
        query_bvecs_si = self.si(query_bvecs)
        latent = self.encoder(torch.cat([observation_bvecs_si, observation_bvals, observation_signals], dim=1), mask)
        signals = self.decoder(latent, query_bvecs_si, query_bvals)

        return latent, signals

    def encoderblock(self, block_id, x, mask):

        for layer_id in range(len(self.encoder_blocks[block_id])):
            x = self.encoder_blocks[block_id][layer_id](x)
            if self.encoder_batchnorm: x = self.encoder_blocks_bn[block_id][layer_id](x)
            x = self.encoder_activation(x)

        local_features = x 
        global_features = torch.amax(local_features-2*mask, dim=2, keepdim=True)

        return local_features, global_features
            
    def encoder(self, x, mask):

        input_points = x.shape[-1]

        for block_id in range(len(self.encoder_blocks)):
            local_features, global_features = self.encoderblock(block_id, x, mask)
            global_features_block = torch.stack(input_points * [global_features], 2)[...,0]
            x = torch.cat([global_features_block, local_features], axis=1)

        return global_features

    def decoder(self, latent, bvecs, bvals):

        query = torch.cat([bvecs, bvals], dim=1)
        x = torch.cat([torch.cat(query.shape[-1]*[latent], dim=2), query], dim=1)
        for idx in range(len(self.decoder_layers)-1):
            x = self.decoder_layers[idx](x)
            if self.decoder_batchnorm: x = self.decoder_layers_bn[idx](x)
            x = self.decoder_activation(x)
        return self.decoder_layers[-1](x)
    
    def si(self, input):

        x = input
        for layer in [self.si_l1, self.si_l2, self.si_l3, self.si_l4]:
            x = self.si_activation(layer(x))
        x1 = x

        x = -input
        for layer in [self.si_l1, self.si_l2, self.si_l3, self.si_l4]:
            x = self.si_activation(layer(x))
        x2 = x
        return x1 + x2