#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute wav2vec embeddings. Modified from original to read filenames from stdin.
"""

import argparse
import glob
import os
from shutil import copy

import sys

import h5py
import soundfile as sf
import numpy as np
import torch
from torch import nn
import tqdm

from fairseq.models.wav2vec import Wav2VecModel


def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3


class PretrainedWav2VecModel(nn.Module):

    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname)
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c


class EmbeddingWriterConfig(argparse.ArgumentParser):

    def __init__(self):
        super().__init__("Pre-compute embeddings for wav2letter++ datasets")

        kwargs = {"action": "store", "type": str, "required": True}

        self.add_argument("--output", "-o",
                          help="Output Directory", **kwargs)
        self.add_argument("--model",
                          help="Path to model checkpoint", **kwargs)
        self.add_argument("--ext", default="wav", required=False,
                          help="Audio file extension")

        self.add_argument("--no-copy-labels", action="store_true",
                          help="Do not copy label files. Useful for large datasets, use --targetdir in wav2letter then.")
        self.add_argument("--use-feat", action="store_true",
                          help="Use the feature vector ('z') instead of context vector ('c') for features")
        self.add_argument("--remove_dims", 
                          help="Path to the file containing a list of dimensions to remove", required=False)

        self.add_argument("--gpu",
                          help="GPU to use", default=0, type=int)


class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()


class H5Writer():
    """ Write features as hdf5 file in wav2letter++ compatible format """

    def __init__(self, fname):
        self.fname = fname
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)

    def write(self, data):
        channel, T = data.shape

        with h5py.File(self.fname, "w") as out_ds:
            data = data.T.flatten()
            out_ds["features"] = data
            out_ds["info"] = np.array([16e3 // 160, T, channel])


class EmbeddingDatasetWriter(object):
    """ Given a model and a wav2letter++ dataset, pre-compute and store embeddings

    Args:
        input_root, str :
            Path to the wav2letter++ dataset
        output_root, str :
            Desired output directory. Will be created if non-existent
    """

    def __init__(self, output_root,                  model_fname,
                 extension="wav",
                 gpu=0,
                 verbose=False,
                 use_feat=False,
                  remove_dims=False,
                 ):

        assert os.path.exists(model_fname)

        self.model_fname = model_fname
        self.model = Prediction(self.model_fname, gpu)

        self.output_root = output_root
        self.verbose = verbose
        self.extension = extension
        self.use_feat = use_feat
        self.remove_dims = remove_dims if len(remove_dims) > 0 else None
        self.paths = list(sys.stdin)
        self.paths = [p.strip() for p in self.paths]

        for path in self.paths:
          assert os.path.exists(path), \
            "Input path '{}' does not exist".format(path)

    def _progress(self, iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        return iterable

    def require_output_path(self, fname=None):
        path = self.get_output_path(fname)
        os.makedirs(path, exist_ok=True)

    @property
    def output_path(self):
        return self.get_output_path()

    def get_output_path(self, fname=None):
        if fname is None:
            return self.output_root
        return os.path.join(self.get_output_path(), fname)

    def copy_labels(self):
        self.require_output_path()

        labels = list(filter(lambda x: self.extension not in x, self.paths))
        for fname in tqdm.tqdm(labels):
            copy(fname, self.output_path)

    def __len__(self):
        return len(self.paths)

    def write_features(self):

        fnames_context = map(lambda x: os.path.join(self.output_path, x.replace("." + self.extension, ".h5context")), \
                             map(os.path.basename, self.paths))
        fnames_context = list(fnames_context)
        feats = []
        nframes = []
        maxframes = 0
        nfiles = 0
        remove_dims = []
        for name, target_fname in self._progress(zip(self.paths, fnames_context), total=len(self)):
            wav, sr = read_audio(name)
            z, c = self.model(wav)
            feat = z if self.use_feat else c
            variance = np.var(feat, axis=1)
            remove_dim = np.argwhere(variance == 0)
#            print(remove_dim.shape)
            for i in range(remove_dim.shape[0]):
              if remove_dim[i,0] not in remove_dims:
                remove_dims.append(remove_dim[i,0])
            feats.append(feat)
            maxframes = max(maxframes, feat.shape[1])
            nframes.append(feat.shape[1])
            nfiles += 1
        feats_arr = np.zeros((nfiles, 512, maxframes))
        for i in range(nfiles):
          feats_arr[i,:, :feats[i].shape[1]] = feats[i]
        feats = feats_arr
#        feats_reshaped = np.reshape(feats_arr, (feats_arr.shape[0] * feats_arr.shape[1], feats_arr.shape[2]))
#        variances = np.var(feats, axis=1)

        if self.remove_dims is None:
          print("Removing dimensions with zero variance");
#          remove_dims = np.argwhere(variances == 0)
          print(remove_dims)
          with open(os.path.join(self.output_path, "remove_dims"), "w") as outfile:
            for dim in remove_dims:
              outfile.write(str(dim))
              outfile.write(" ")
        else:
          print("Removing dimensions specified in %s" % self.remove_dims)
          with open(self.remove_dims, "r") as infile:
            remove_dims = [int(x) for x in infile.read().split(" ") if len(x) > 0 ]
#          i  remove_dims = np.array(remove_dims)
#        print("Dimensions : " + str(remove_dims))
#        print(feats.shape)
        feats = np.delete(feats, remove_dims, axis=1)
        print(feats.shape)
#        for dim in remove_dims:
#          feats[:,dim,:] = np.random.normal(size=feats.shape[2])
#          feats = np.delete(feats, remove_dims, axis=1)
        
#        u, s, v = np.linalig.svd(feats, full_matrices=True)
#        print(s.shape)
#        variances = np.var(feats, 1)
        #feats = feats[:, :50, :]
        for i, target_fname in enumerate(fnames_context):
          writer = H5Writer(target_fname)
          writer.write(feats[i,:,:nframes[i]])
          
    def __repr__(self):

        return "EmbeddingDatasetWriter ({n_files} files)\n\tinput:\t{paths}\n\toutput:\t{output_root}\n\t)".format(
            n_files=len(self), **self.__dict__)


if __name__ == "__main__":

    args = EmbeddingWriterConfig().parse_args()

    writer = EmbeddingDatasetWriter(
            output_root=args.output,
            model_fname=args.model,
            gpu=args.gpu,
            extension=args.ext,
            use_feat=args.use_feat,
            remove_dims=args.remove_dims
        )

#    print(writer)
    writer.require_output_path()

 #   print("Writing Features...")
    writer.write_features()
  #  print("Done.")

    if not args.no_copy_labels:
        #print("Copying label data...")
        writer.copy_labels()
        #print("Done.")
