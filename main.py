import pandas as pd
import numpy as np
from tqdm import tqdm
import RBF_Net
import pipeline as pl
import score_and_plots as sap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


if __name__ == '__main__':

    df = pd.read_csv(r'random_spheres', delimiter='\t')

    mon_len = 2
    mon_par = df.columns[:mon_len]
    print(mon_par)

    x, y = np.array(df)[:, mon_len:], np.array(df)[:, :mon_len]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, shuffle=True)

    pipe1 = pl.PipeLine().load_data(x_train, y_train, [0, 180]).weighting('m_func').cutter([10, 70], [10, 70])  #
    pipe2 = pl.PipeLine().load_data(x_test, y_test, [0, 180]).weighting('m_func').cutter([10, 70], [10, 70])  #

    pipe1.noise_gen(forward_sigma=0.01,  replicas=0, noise_type='additive_fixed')
    pipe2.noise_gen(forward_sigma=0.01,  replicas=0, noise_type='additive_fixed')

    train_x, train_y = pipe1.get_data()[0], pipe1.get_data()[2]
    test_x, test_y = pipe2.get_data()[0], pipe2.get_data()[2]

    double_layer_RBF = RBF_Net.DRBF_Net(N1 = 20, N2 = 500, dist_type='median', norm_type='L2')
    double_layer_RBF.fit(train_x, train_y)
    y_pr = double_layer_RBF.predict(test_x)

    sap.predict_regr_plot(test_y, y_pr, mon_par, '')
    sap.error_regr_plot(test_y, y_pr, mon_par, '')


    rbf_net = RBF_Net.RBFNet(norm_type='L2', dist_type='std')
    rbf_net.fit(train_x, train_y, n_centers = 1500)
    y_predict = rbf_net.predict(test_x)
    sap.predict_regr_plot(test_y, y_predict, mon_par, '')
    sap.error_regr_plot(test_y, y_predict, mon_par, '')
