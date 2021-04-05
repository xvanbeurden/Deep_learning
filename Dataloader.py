from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

root_dir = r'gdrive/MyDrive/Datafiles_mat/'
root_dir_images = r'gdrive/MyDrive/Datasets/'

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_database(path):
  mat = loadmat(path)
  matStruct = mat['dbStruct'].item()

  whichSet = matStruct[0].item()
  dataset = 'pittsburgh'
  dbImage = [f[0].item() for f in matStruct[1]]
  utmDb = matStruct[2].T

  qImage = [f[0].item() for f in matStruct[3]]
  utmQ = matStruct[4].T

  numDb = matStruct[5].item()
  numQ = matStruct[6].item()

  posDistThr = matStruct[7].item()
  posDistSqThr = matStruct[8].item()
  nonTrivPosDistSqThr = matStruct[9].item()

  return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
          utmQ, numDb, numQ, posDistThr,
          posDistSqThr, nonTrivPosDistSqThr)


class DatasetImages(Dataset):
    def __init__(self, structFile, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.dbStruct = parse_database(structFile)
        self.dataset = self.dbStruct.dataset

        self.images = [join(root_dir_images, dbIm) for dbIm in self.dbStruct.dbImage]

    def __getitem__(self, index):
        image = Image.open(self.images[index])

        # if self.input_transform:
        #     image = self.input_transform(image)

        return image, index

    def __len__(self):
        return len(self.images)

    def getPositivesForEval(self):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        self.distances, self.pot_positives_eval = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                       radius=self.dbStruct.posDistThr)

        return self.pot_positives_eval


class QueryImages(Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, nPos=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.dbStruct = parse_database(structFile)

        self.margin = margin
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nNeg = nNeg  # number of negatives used for training

        self.images = [join(root_dir_images, dbIm) for dbIm in dbImage]

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        self.distances, self.pot_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.nonTrivPosDistSqThr ** 0.5)

        for i, posi in enumerate(self.pot_positives):
            self.pot_positives[i] = np.sort(posi)

        self.hard_negatives = []
        self.distancesforNeg, self.positivesforNeg = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                          radius=self.dbStruct.posDistThr)
        for i in self.positivesforNeg:
            self.hard_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), i, assume_unique=True))

        self.queries = np.where(np.array([len(x) for x in self.pot_positives]) > 0)[0]

    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index + qOffset]

            posFeat = h5feat[self.pot_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1)  # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)
            dPos = dPos.item()
            posIndex = self.pot_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.hard_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1, -1),
                                         self.nNeg * 10)  # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin ** 0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        # if self.input_transform:
        #     query = self.input_transform(query)
        #     positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

        def __len__(self):
            return len(self.queries)


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_training_dataset():
  file_dir = join(root_dir, 'pitts250k_train.mat')
  return DatasetImages(file_dir)

def get_val_dataset():
  file_dir = join(root_dir, 'pitts250k_val.mat')
  return DatasetImages(file_dir)

def get_test_dataset():
  file_dir = join(root_dir, 'pitts250k_test.mat')
  return DatasetImages(file_dir)

def get_training_query_dataset():
  file_dir = join(root_dir, 'pitts250k_train.mat')
  return QueryImages(file_dir)

def get_val_query_dataset():
  file_dir = join(root_dir, 'pitts250k_val.mat')
  return QueryImages(file_dir)

def get_test_query_dataset():
  file_dir = join(root_dir, 'pitts250k_test.mat')
  return QueryImages(file_dir)