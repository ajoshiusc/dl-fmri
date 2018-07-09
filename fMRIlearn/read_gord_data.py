#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""

import nilearn as nl
import os
import glob
import scipy as sp
from scipy.io import loadmat
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Imputer


class bfpData():
    """ This class manages BFP data set"""

    def __init__(self):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.nvert_hemi = 32492
        self.nvert_subcort = 31870
        self.data_dir = None
        self.subids = None
        self.data = None
        self.dirlst = None
        print("Read flat maps for left and right hemispheres.")

    def get_data(self):
        """get the rfMRI data"""
        return self.data, self.cog_scores, self.subids

    def read_fmri(self, data_dir, reduce_dim=None, int_subid=1):
        """ Read fMRI data from disk
            If reduce_dim = None, no dimesionality reduction is performed"""
        self.data_dir = data_dir
        self.dirlst = glob.glob(self.data_dir + '/*.mat')
        self.data = list()
        self.subids = list()

        print('\n=======\n Reading fMRI data\n=======\n')

        if reduce_dim is not None:
            pca = PCA(n_components=reduce_dim)

        for subfile in self.dirlst:
            subid = subfile.replace('_rest_bold.32k.GOrd.mat', '')
            subid = subid.replace(self.data_dir + '/', '')

            outfile = self.data_dir + '/processed/' + subid + 'pca_reduced.npz'

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

                    # Save the data for faster processing
                    sp.savez_compressed(outfile, fmri_data=fmri_data)

            self.subids.append(subid)
            self.data.append(fmri_data)

    def read_cog_scores(self, cogscore_file):
        """ Read cognitive scores from csv file """
        self.cog_scores = pd.read_csv(cogscore_file, index_col=0)
        ''' If fMRI data exists for some subjects, then store their cognitive scores '''
        for subid in self.subids:
            self.cog_scores.append(self.get_cog_score_subid(subid))

    def get_cog_score_subid(self, subid):
        """ Get cognitive score for a given subject id"""
        return self.cog_scores.loc[subid]
