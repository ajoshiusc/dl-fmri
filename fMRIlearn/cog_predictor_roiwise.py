#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from fMRIlearn.read_gord_data import bfpData
from fMRIlearn.brainsync import brainSync
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
# Define cognitive predictor regressor This takes fMRI grayordinate data as
# input and cognitive scores as output


class CogPredRoiwise(BaseEstimator, bfpData):
    '''This takes fMRI grayordinate data as input and cognitive scores as output'''

    def __init__(self, bfp_dir):
        # read the flat maps for both hemispheres
        bfpData.__init__(self, bfp_dir)

    def train_model_roiwise(self, data_dir, csv_file):
        print('Fitting the model')
        self.read_fmri(data_dir, roiwise=1)
        self.read_cog_scores(csv_file)

        Xlst, y, subids = self.get_data()

        y = self.cog_scores['Performance IQ'][self.subids].get_values()

        nsub = len(Xlst)
        udiag_ind = np.triu_indices(Xlst[0].shape[0], k=1)
        n_conn = len(udiag_ind[0])
        print(udiag_ind)
        X = np.zeros((nsub, n_conn))
        print(nsub, X.shape)

        for subno in range(nsub):

            Conn = np.corrcoef(Xlst[subno].T)
            print('shape of conn is :', Conn.shape)
            X[subno, :] = Conn[udiag_ind]

            print(X.shape, y.shape)

        sc = StandardScaler()
        X = sc.fit_transform(X)
#        y = sc.fit_transform(y.reshape(1, -1))

        inputs = Input(shape=(n_conn, ))
        preds = Dense(1, activation='linear')(inputs)

        model = Model(inputs=inputs, outputs=preds)
        sgd = SGD(lr=1e-5)
        model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
        model.fit(X, y.T, batch_size=5, epochs=300, shuffle=False, validation_split=0.2)
