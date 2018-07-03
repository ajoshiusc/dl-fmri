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
train_data_dir = '/deneb_disk/ADHD_Peking_bfp/training'
test_data_dir = '/deneb_disk/ADHD_Peking_bfp/testing'

csv_file = '/deneb_disk/ADHD_Peking_bfp/Peking_all_phenotypic.csv'

cp = cpred.CogPred(bfp_dir)

cp.train_model(data_dir=train_data_dir, csv_file=csv_file)
y, ypred = cp.predict(data_dir=test_data_dir, csv_file=csv_file)

print(y, ypred)

print('Model Trained')


# if __name__ == "__main__":
#    main()
