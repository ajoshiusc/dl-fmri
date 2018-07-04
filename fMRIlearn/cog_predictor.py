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
from keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Flatten, Dense
from keras.models import Model
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
        self.ref_subno = 0

    def map_gord2sqrs(self, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """
        print(sqr_size)
        x_ind, y_ind = np.meshgrid(
            np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, sqr_size),
            np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, sqr_size))

        print(x_ind.shape)
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
                sqr_inds = (x_ind, y_ind)
                sqr_data_left[subn, :, :, t_ind] = griddata(
                    self.sqrmap, lh_data, sqr_inds)
                # Map Right Hemisphere data
                rh_data = data[self.nvert_hemi:2 * self.nvert_hemi, t_ind]
                rh_data = rh_data[self.sqr_map_ind]
                sqr_data_right[subn, :, :, t_ind] = griddata(
                    self.sqrmap, rh_data, sqr_inds)
                print(str(t_ind) + ',', end='', flush=True)

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
            sub2s = brainSync(sub1.T, sub2.T)
            dist_mat[sub1no, sub2no] = np.linalg.norm(sub1, sub2s)
            print(sub1no, sub2no)

        self.ref_subno = np.argmax(np.sum(dist_mat, axis=1))
        print('The most representative subject is %d' % self.ref_subno)

    def sync2rep(self):
        """ Sync all subjects to the representative subject """

        ref = self.data[self.ref_subno]
        ref = StandardScaler().fit_transform(ref.T)

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
            'weights3d.h5', monitor='val_loss', save_best_only=True)

        X = self.nn_ipdata
        #        y=np.array([11,12,13,14,15]).reshape((5,1))
        y = self.cog_scores['ADHD Index'][self.subids].get_values()

        X[0] = X[0][y != -999, :, :, :]
        X[1] = X[1][y != -999, :, :, :]
        X[2] = X[2][y != -999, :, :]
        y = y[y != -999]
        # y = y[:]

        print('Number of subjects: %d\n' % (y.shape[0]))

        history = self.hybrid_cnn.fit(
            X,
            y,
            batch_size=5,
            epochs=20,
            verbose=1,
            shuffle=True,
            validation_split=0.2,
            callbacks=[model_checkpoint])

        print('=======\nSaving training history\n=======')
        with open('/trainHistoryDict', 'wb') as file_pi:
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
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def predict(self, data_dir, csv_file):

        mod = self.get_neural_net()

        mod.load_weights('weights3d.h5')

        self.read_fmri(data_dir, reduce_dim=21)
        self.read_cog_scores(csv_file)
        self.map_gord2sqrs()
        X = self.nn_ipdata
        y = self.cog_scores['ADHD Index'][self.subids].get_values()

        ypred = mod.predict(X, verbose=1)

        return y, ypred

    def get_neural_net(self, isize=[256, 256]):
        input_lh = Input((isize[0], isize[1], 21))
        input_rh = Input((isize[0], isize[1], 21))
        sub_cort = Input((self.nvert_subcort, 21))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_lh)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_rh)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        d1_ip = Dense(512, activation='relu')(sub_cort)
        d1_ip = Flatten()(d1_ip)
        d1_ip = Dense(1, activation='relu')(d1_ip)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_1)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        flat1 = Flatten()(conv5)

        d1 = Dense(512, activation='relu')(flat1)
        d2 = Dense(1, activation='relu')(d1)
        d2 = concatenate([d2, d1_ip], axis=-1)

        out_theta = Dense(1)(d2)
        #    conv_tx = Conv2D(1, (1, 1), activation=final_activation)(conv5)
        #    conv_ty = Conv2D(1, (1, 1), activation=final_activation)(conv5)
        #    conv_theta = Conv2D(1, (1, 1), activation='tanh')(conv5)

        #    out_img = rotate(inputs,conv_theta)

        #        model = Model(inputs=[input_lh, input_rh, sub_cort], outputs=out_theta)
        model = Model(inputs=[input_lh, input_rh, sub_cort], outputs=out_theta)

        model.compile(
            optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])

        return model


# fdfdh
