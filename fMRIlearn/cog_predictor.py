#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import os
import pickle
import itertools
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import griddata
from keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Flatten, Dense, ZeroPadding2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import losses
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from fMRIlearn.read_gord_data import bfpData
from fMRIlearn.brainsync import brainSync

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# Define cognitive predictor regressor This takes fMRI grayordinate data as
# input and cognitive scores as output


class CogPred(BaseEstimator, bfpData):
    '''This takes fMRI grayordinate data as input and cognitive scores as output'''

    def __init__(self, bfp_dir):
        # read the flat maps for both hemispheres
        bfpData.__init__(self)
        dat = loadmat(os.path.join(bfp_dir, 'supp_data', 'sqrmap.mat'))
        self.sqrmap = dat['sqrmap']
        # for converting indices from matlab to python, subtract -1
        self.sqr_map_ind = dat['data_ind'] - 1
        self.sqr_map_ind = self.sqr_map_ind.squeeze()
        self.sqr_data = list()
        self.nn_ipdata = list()
        print("Read flat maps for left and right hemispheres.")
        self.hybrid_cnn = self.get_neural_net()
        self.ref_subno = None
        self.ref_data = None

    def map_gord2sqrs(self, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """
        x_ind, y_ind = np.meshgrid(
            np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, sqr_size),
            np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, sqr_size))
        sqr_inds = (x_ind, y_ind)

        print('\n=======\n Mapping to square\n=======\n')

        sqr_data_sz = (len(self.data), x_ind.shape[0], y_ind.shape[1],
                       self.data[0].shape[1])
        sqr_data_left = np.zeros(sqr_data_sz)
        sqr_data_right = np.zeros(sqr_data_sz)
        noncortical_data = np.zeros((len(self.data), self.nvert_subcort,
                                     self.data[0].shape[1]))
        subn = 0
        for data in self.data:
            for t_ind in np.arange(data.shape[1]):

                # Map Left Hemisphere data
                lh_data = data[:self.nvert_hemi, t_ind][self.sqr_map_ind]
                sqr_data_left[subn, :, :, t_ind] = griddata(
                    self.sqrmap, lh_data, sqr_inds)

                # Map Right Hemisphere data
                rh_data = data[self.nvert_hemi:2 * self.nvert_hemi, t_ind]
                rh_data = rh_data[self.sqr_map_ind]

                sqr_data_right[subn, :, :, t_ind] = griddata(
                    self.sqrmap, rh_data, sqr_inds)
                print('.', end='', flush=True)

            print('subno = %d' % subn)

            noncortical_data[subn, :, :] = np.nan_to_num(
                data[2 * self.nvert_hemi:, ])
            self.data[subn] = 0

            sqr_data_right = np.nan_to_num(sqr_data_right)
            sqr_data_left = np.nan_to_num(sqr_data_left)

            subn += 1

        self.nn_ipdata = [sqr_data_left, sqr_data_right, noncortical_data]

        return self.nn_ipdata

    def choose_rep(self):
        """Choses representative subject to be used as target for BrainSync"""
        nsub = len(self.subids)
        subs = range(nsub)
        dist_mat = np.zeros((nsub, nsub))

        for sub1no, sub2no in itertools.product(subs, subs):
            sub1 = self.data[sub1no]
            sub2 = self.data[sub2no]
            sub1 = StandardScaler().fit_transform(sub1.T)
            sub2 = StandardScaler().fit_transform(sub2.T)  # .T to make it TxV
            sub2s, _ = brainSync(sub1, sub2)
            dist_mat[sub1no, sub2no] = np.linalg.norm(sub1.flatten() -
                                                      sub2s.flatten())
            print(sub1no, sub2no)

        self.ref_subno = np.argmin(np.sum(dist_mat, axis=1))
        self.ref_data = self.data[self.ref_subno]

        # Save the reference subject and ref subject data
        np.savez_compressed(
            'Refdata_test.npz',
            ref_data=self.ref_data,
            ref_subno=self.ref_subno)

        print('The most representative subject is %d' % self.ref_subno)

    def sync2rep(self):
        """ Sync all subjects to the representative subject """
        if self.ref_subno is None:
            print(
                '=======\n Reference subject is not initialized, loading from a file\n=======\n'
            )
            a = np.load('Refdata_test.npz')
            self.ref_subno = a['ref_subno']
            self.ref_data = a['ref_data']

        ref = StandardScaler().fit_transform(self.ref_data.T)

        for subno in range(len(self.subids)):
            sub = self.data[subno]
            sub = StandardScaler().fit_transform(sub.T)  # .T to make data TxV
            sub_sync, _ = brainSync(ref, sub)
            self.data[subno] = sub_sync.T

    def train_model(self, data_dir, csv_file):
        """ X: data in grayordinates of shape Vert x Time x Subj
            y: cognitive scores"""
        #     self.map_gord2sqrs(X)
        print('Fitting the model')

        self.read_fmri(data_dir, reduce_dim=21)
        self.choose_rep()
        self.sync2rep()
        self.read_cog_scores(csv_file)
        self.map_gord2sqrs()

        model_checkpoint = ModelCheckpoint(
            'weights3d_test.h5', monitor='val_loss', save_best_only=True)

        X = self.nn_ipdata
        #        y=np.array([11,12,13,14,15]).reshape((5,1))
        y = self.cog_scores['ADHD Index'][self.subids].get_values()

        X[0] = X[0][y > 0, :, :, :]
        X[1] = X[1][y > 0, :, :, :]
        X[2] = X[2][y > 0, :, :]
        y = y[y > 0] / 50.0
        # y = y[:]
        X = X[0].astype('float32')
        y = y.astype('float32')

        print('training with this data\n')
        print(y)

        history = self.hybrid_cnn.fit(
            X,
            y,
            batch_size=10,
            epochs=20,
            verbose=1,
            shuffle=True,
            validation_split=0.2,
            callbacks=[model_checkpoint])

        print('=======\nSaving training history\n=======')
        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        print('=======\nDisplaying training history\n=======')
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('model fit mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def predict(self, data_dir, csv_file):

        mod = self.get_neural_net()
        mod.load_weights('weights3d_test.h5')

        self.read_fmri(data_dir, reduce_dim=21)
        self.sync2rep()
        self.read_cog_scores(csv_file)
        self.map_gord2sqrs()
        X = self.nn_ipdata[0].astype('float32')
        y = self.cog_scores['ADHD Index'][self.subids].get_values() / 50.0
        y = y.astype('float32')
        ypred = mod.predict(X, verbose=1)

        return y, ypred

    def get_neural_net(self, isize=[256, 256]):
        """VGG model with one FC layer added at the end for continuous output"""

        img_rows = isize[0]
        img_cols = isize[1]
        #VGG model
        main_input = Input(
            shape=(isize[0], isize[1], 21), dtype='float32', name='main_input')
        zpad1 = ZeroPadding2D(
            (1, 1), input_shape=(isize[0], isize[1], 21),
            name='zpad1')(main_input)
        conv1 = Conv2D(64, (3, 3), name='conv1', activation='relu')(zpad1)
        zpad2 = ZeroPadding2D((1, 1), name='zpad2')(conv1)
        conv2 = Conv2D(64, (3, 3), name='conv2', activation='relu')(zpad2)
        maxp1 = MaxPooling2D((2, 2), strides=(2, 2), name='maxp1')(conv2)

        zpad3 = ZeroPadding2D((1, 1), name='zpad3')(maxp1)
        conv3 = Conv2D(128, (3, 3), name='conv3', activation='relu')(zpad3)
        zpad4 = ZeroPadding2D((1, 1), name='zpad4')(conv3)
        conv4 = Conv2D(128, (3, 3), name='conv4', activation='relu')(zpad4)
        maxp2 = MaxPooling2D((2, 2), strides=(2, 2), name='maxp2')(conv4)

        zpad5 = ZeroPadding2D((1, 1), name='zpad5')(maxp2)
        conv5 = Conv2D(256, (3, 3), name='conv5', activation='relu')(zpad5)
        zpad6 = ZeroPadding2D((1, 1), name='zpad6')(conv5)
        conv6 = Conv2D(256, (3, 3), name='conv6', activation='relu')(zpad6)
        zpad7 = ZeroPadding2D((1, 1), name='zpad7')(conv6)
        conv7 = Conv2D(256, (3, 3), name='conv7', activation='relu')(zpad7)
        maxp3 = MaxPooling2D((2, 2), strides=(2, 2), name='maxp3')(conv7)

        zpad8 = ZeroPadding2D((1, 1), name='zpad8')(maxp3)
        conv8 = Conv2D(512, (3, 3), name='conv8', activation='relu')(zpad8)
        zpad9 = ZeroPadding2D((1, 1), name='zpad9')(conv8)
        conv9 = Conv2D(512, (3, 3), name='conv9', activation='relu')(zpad9)
        zpad10 = ZeroPadding2D((1, 1), name='zpad10')(conv9)
        conv10 = Conv2D(512, (3, 3), name='conv10', activation='relu')(zpad10)
        maxp4 = MaxPooling2D((2, 2), strides=(2, 2), name='maxp4')(conv10)

        zpad11 = ZeroPadding2D((1, 1), name='zpad11')(maxp4)
        conv11 = Conv2D(512, (3, 3), name='conv11', activation='relu')(zpad11)
        zpad12 = ZeroPadding2D((1, 1), name='zpad12')(conv11)
        conv12 = Conv2D(512, (3, 3), name='conv12', activation='relu')(zpad12)
        zpad13 = ZeroPadding2D((1, 1), name='zpad13')(conv12)
        conv13 = Conv2D(512, (3, 3), name='conv13', activation='relu')(zpad13)
        maxp5 = MaxPooling2D((2, 2), strides=(2, 2), name='maxp5')(conv13)

        flatten = Flatten(name='flatten')(maxp5)
        dense1 = Dense(4096, activation='relu', name='dense1')(flatten)
        dropout1 = Dropout(0.5, name='dropout1')(dense1)
        dense2 = Dense(4096, activation='relu', name='dense2')(dropout1)
        dropout2 = Dropout(0.5, name='dropout2')(dense2)
        dense3 = Dense(1000, activation='relu', name='dense3')(dropout2)
        dropout3 = Dropout(0.5, name='dropout3')(dense3)
        dense4 = Dense(256, activation='relu', name='dense4')(dropout3)

        # My addition of regression layer
        out_theta = Dense(1)(dense4)
        print("==Defining Model  ==")
        model = Model(inputs=[main_input], outputs=[out_theta])
        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(
            optimizer=sgd, loss=losses.mean_squared_error, metrics=['mse'])

        return model
