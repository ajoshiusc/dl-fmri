#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""

import nilearn as nl
import os


class HCPData():
    """ This class manages HCP data"""

    def __init__(self, hcp_dir):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.hcp_dir = hcp_dir
        print("Read flat maps for left and right hemispheres.")

    def get_data(self):
        """get the rfMRI data"""
        self.read_fMRI()
    #    self.read_cog_scores(self.subids)
        return self.rfmri, self.cog_scores
    
    def read_fMRI(self):
        """ Read fMRI data from disk """
        dirlst = os.listdir(self.hcp_dir)

        for subid in dirlst:
            print(subid)

