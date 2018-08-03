#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import numpy as np
import os
import glob
import scipy as sp
from scipy.io import loadmat
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Imputer


def roiwise_timeseries(fmri_data, gord_label):
    """ fmri_data : nT x nVert"""
    lab_list = np.unique(gord_label)
    lab_list = np.setdiff1d(lab_list, [0])  # remove background roi
    lab_list = lab_list[~np.isnan(lab_list)]  # remove nans

    n_rois = len(lab_list)
    n_time = fmri_data.shape[1]

    out_data = np.zeros((n_rois, n_time))

    for roino, roiid in enumerate(lab_list):
        out_data[roino, :] = sp.stats.trim_mean(
            fmri_data[gord_label == roiid, :], proportiontocut=.1, axis=0)

    return out_data


class bfpData():
    """ This class manages BFP data set"""

    def __init__(self, bfp_dir):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.nvert_hemi = 32492
        self.nvert_subcort = 31870
        self.data_dir = None
        self.subids = None
        self.fmri_data = list()
        self.shape_data = list()
        self.data = list()
        self.dirlst = None
        self.data_dir_sct = list()
        self.data_dir_fmri = list()
        print("Read flat maps for left and right hemispheres.")
        dat = loadmat(
            os.path.join(bfp_dir, 'supp_data', 'USCBrain_grayord_labels.mat'))
        self.gord_label = dat['labels'].squeeze()

    def get_data(self):
        """get the rfMRI data"""
        return self.fmri_data, self.shape_data, self.cog_scores, self.subids

    def read_shape(self, data_dir):
        """ Read SCT anatomical data from disk
            Data is SCT multires x vertices x sub """
        self.data_dir_sct = data_dir

        for subid in self.subids:
            sctfile = self.data_dir_sct + '/' + str(subid) + '_T1w.SCT.GOrd.mat'
            a = loadmat(sctfile)
            sct = a['SCT_GO']
            print('Read sub SCT = ' + str(subid))
            # Preprocess fMRI, replace Nan by avg of cortical activity at
            # that time point and standardize this should be interesting
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            sct = imp.fit_transform(sct)
            sct = StandardScaler().fit_transform(sct)

            self.shape_data.append(sct)

    def concat_shape_fmri(self):
        """ Concatenate fmri and shape data """
        for ind in range(len(self.subids)):
            full_data = np.concatenate(
                (self.fmri_data[ind], self.shape_data[ind]), axis=1)
            self.data.append(full_data)
            self.fmri_data[ind] = None
            self.shape_data[ind] = None
            
        self.fmri_data = list()
        self.shape_data = list()



    def read_fmri(self, data_dir, reduce_dim=None, int_subid=1, roiwise=1):
        """ Read fMRI data from disk
            If reduce_dim = None, no dimesionality reduction is performed
            data is Time x Vertices x sub"""
        self.data_dir_fmri = data_dir
        self.dirlst = glob.glob(self.data_dir_fmri + '/*.mat')
        self.fmri_data = list()
        self.subids = list()

        print('\n=======\n Reading fMRI data\n=======\n')

        if reduce_dim is not None:
            pca = PCA(n_components=reduce_dim)

        for subfile in self.dirlst:
            subid = subfile.replace('_rest_bold.32k.GOrd.mat', '')
            subid = subid.replace(self.data_dir_fmri + '/', '')

            outfile = self.data_dir_fmri + '/processed/' + subid + 'pca_reduced.npz'

            if int_subid:
                subid = int(subid)

            if os.path.isfile(outfile):
                a = sp.load(outfile)
                fmri_data = a['fmri_data']
                print('Fast Read sub id = ' + str(subid))

            else:
                if os.path.isfile(subfile):
                    print('Reading sub id = ' + str(subid))
                    fmri_data = loadmat(subfile)['dtseries']

                    # Preprocess fMRI, replace Nan by avg of cortical activity at
                    # that time point and standardize this should be interesting
                    imp = Imputer(
                        missing_values='NaN', strategy='mean', axis=0)
                    fmri_data = imp.fit_transform(fmri_data)
                    fmri_data = StandardScaler().fit_transform(fmri_data)

                    if reduce_dim != None:
                        fmri_data = pca.fit_transform(fmri_data)
                        sp.savez_compressed(outfile, fmri_data=fmri_data)

                    if roiwise > 0:
                        fmri_data = roiwise_timeseries(fmri_data,
                                                       self.gord_label)

            self.subids.append(subid)
            self.fmri_data.append(fmri_data)

    def read_cog_scores(self, cogscore_file):
        """ Read cognitive scores from csv file """
        self.cog_scores = pd.read_csv(cogscore_file, index_col=0)
        ''' If fMRI data exists for some subjects, then store their cognitive scores '''
        for subid in self.subids:
            self.cog_scores.append(self.get_cog_score_subid(subid))

    def get_cog_score_subid(self, subid):
        """ Get cognitive score for a given subject id"""
        return self.cog_scores.loc[subid]
