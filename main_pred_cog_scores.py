#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

#||AUM||
#||Shree Ganeshaya Namaha||
import fMRIlearn.cog_predictor as cpred
#import fMRIlearn.read_gord_data as rh

# def main():
""" Main script that calls the functions object"""
bfp_dir = '/home/ajoshi/coding_ground/bfp'
data_dir = '/deneb_disk/temp1' #ADHD_Peking_bfp'
csv_file = '/deneb_disk/ADHD_Peking_bfp/Peking_all_phenotypic.csv'

cp = cpred.CogPred(bfp_dir)

nn = cp.get_neural_net()

cp.read_fmri(data_dir, reduce_dim=21)
cp.read_cog_scores(csv_file)
cp.map_gord2sqrs()
cp.train_model()

print('Model Trained')


# if __name__ == "__main__":
#    main()
