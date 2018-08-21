#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

# ||AUM||
# ||Shree Ganeshaya Namaha||
import numpy as np
import fMRIlearn.cog_predictor as cpred


def main():
    """ Main script that calls the functions object"""
    bfp_dir = '/big_disk/ajoshi/coding_ground/bfp'
    train_data_dir = '/big_disk/ajoshi/ADHD_Peking_bfp/training'  #ADHD_Peking_bfp/training'
    test_data_dir = '/big_disk/ajoshi/ADHD_Peking_bfp/testing'

    csv_file = '/big_disk/ajoshi/ADHD_Peking_bfp/Peking_all_phenotypic.csv'

    cp = cpred.CogPred(bfp_dir)

    cp.train_model(data_dir=train_data_dir, csv_file=csv_file)
    y, ypred = cp.predict(data_dir=test_data_dir, csv_file=csv_file)

    np.savez_compressed('pred_res.npz', y=y, ypred=ypred)

    print(y[:, 0], ypred[:, 0])
    print(y[:, 1], ypred[:, 1])
    print(y[:, 2], ypred[:, 2])

    rowind = y[:, 2] > 0
    print('Correlation between predicted and actual values: ')
    print(np.corrcoef(y[rowind, 0].T, ypred[rowind, 0].T)[0, 1])
    print(np.corrcoef(y[rowind, 1].T, ypred[rowind, 1].T)[0, 1])
    print(np.corrcoef(y[rowind, 2].T, ypred[rowind, 2].T)[0, 1])


if __name__ == "__main__":
    main()
