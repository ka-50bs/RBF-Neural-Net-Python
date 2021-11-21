import pandas as pd
import numpy as np
from tqdm import tqdm
import score_and_plots as sap
import matplotlib.pyplot as plt

if __name__ == '__main__':

    y_pred = np.loadtxt('Single_layer_y_predict.txt')
    y_test = np.loadtxt('Single_layer_y_test.txt')

    # y_pred = np.loadtxt('y_predict.txt')
    # y_test = np.loadtxt('y_test.txt')

    # y_pred = np.loadtxt('Single_layer_y_predict_sph.txt')
    # y_test = np.loadtxt('Single_layer_y_test_sph.txt')


    # y_pred = np.loadtxt('y_predict_shhh.txt')
    # y_test = np.loadtxt('y_test_shhh.txt')

    mon_par = ['r', 'ε','n', 'ψ']
    # mon_par = ['r', 'n']

    sap.predict_regr_plot(y_test, y_pred, mon_par,
                          'Бесшумный случай регулярная БД, весовая функция - м-функция, 1500 центров')
    sap.error_regr_plot(y_test, y_pred, mon_par,
                        'Бесшумный случай регулярная БД, весовая функция - м-функция, 1500 центров')

