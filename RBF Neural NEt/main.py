import pandas as pd
import numpy as np
import RBF_Net
import pipeline as pl
import score_and_plots as sap
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df_train = pd.read_csv(r'D:\RBF\random_spheres_train.csv', delimiter=';')
    df_test = pd.read_csv(r'D:\RBF\random_spheres_test.csv', delimiter=';')

    #df_dim = pd.read_csv(r'D:\dimer_bd_5000.csv', delimiter=',')
    #df_mon = pd.read_csv(r'D:\Plt_200k_h2.csv', delimiter=',')

    y_train = np.array(df_train)[:, :2]
    x_train = (np.array(df_train)[:, 2:])

    y_test = np.array(df_test)[:, :2]
    x_test = (np.array(df_test)[:, 2:])

    x_train = pl.weighting(x_train, 'm_func', [0, 180])
    x_test = pl.weighting(x_test, 'm_func', [0, 180])

    x_train = pl.cutter(x_train, 10, 70)
    x_test = pl.cutter(x_test, 10, 70)

    y_train_noise = np.copy(y_train)
    x_train_noise = np.copy(x_train)
    for k in range(4):
        x_train_buf = np.copy(x_train)
        for i in range(len(x_train_buf)):
            sigma = np.max((x_train_buf[i])) * 0.03
            for j in range(len(x_train_buf[i])):
                x_train_buf[i][j] = np.random.normal(x_train_buf[i][j], sigma)
        x_train_noise = np.vstack([x_train_noise, x_train_buf])
        y_train_noise = np.vstack([y_train_noise, y_train])

    rbf = RBF_Net.RBFNet()
    rbf.fit(x_train_noise, y_train_noise, 2000)

    ypr = rbf.predict(x_test)
    params = ['r', 'n']
    sap.predict_plot(y_test, ypr, params)
    sap.error_plot(y_test, ypr, params)

    x_test_noise = np.copy(x_test)
    for i in range(len(x_test_noise)):
        sigma = np.max((x_test_noise[i])) * 0.03
        for j in range(len(x_test_noise[i])):
            x_test_noise[i][j] = np.random.normal(x_test_noise[i][j], sigma)

    ypr = rbf.predict(x_test_noise)
    params = ['r', 'n']
    sap.predict_plot(y_test, ypr, params)
    sap.error_plot(y_test, ypr, params)
    plt.plot(x_test_noise[1000])
    plt.show()