import pandas as pd
import numpy as np
import RBF_Net
import pipeline as pl
import score_and_plots as sap
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df_train = pd.read_csv(r'D:\RBF\random_spheres_train.csv', delimiter=';')
    df_test = pd.read_csv(r'D:\RBF\random_spheres_test.csv', delimiter=';')

    y_train = np.array(df_train)[:, :2]
    x_train = (np.array(df_train)[:, 2:])

    y_test = np.array(df_test)[:, :2]
    x_test = (np.array(df_test)[:, 2:])

    pipe0 = pl.PipeLine()
    pipe0.load_data(x_test, y_test, [0, 180])
    pipe0.weighting('m_func')
    pipe0.cutter([10, 70])
    x_test_clear, y_test_clear = pipe0.get_data()

    pipe1 = pl.PipeLine()
    pipe1.load_data(x_train, y_train, [0, 180])
    pipe1.weighting('m_func')
    pipe1.cutter([10, 70])
    pipe1.noise_generation(0.03, 3)
    x_train_noise, y_train_noise = pipe1.get_data()

    pipe2 = pl.PipeLine()
    pipe2.load_data(x_test, y_test, [0, 180])
    pipe2.weighting('m_func')
    pipe2.cutter([10, 70])
    pipe2.noise_generation(0.03, 0)
    x_test_noise, y_test_noise = pipe2.get_data()

    rbf = RBF_Net.RBFNet()
    rbf.fit(x_train_noise, y_train_noise, 1000)

    ypr = rbf.predict(x_test_clear)
    params = ['r', 'n']
    sap.predict_plot(y_test_clear, ypr, params, 'Предсказание для безшумных индикатрис. \nОбучающая выборка с шумом 3%.')
    sap.error_plot(y_test_clear, ypr, params, 'Ошибка предсказания для безшумных индикатрис. \nОбучающая выборка с шумом 3%.')

    ypr = rbf.predict(x_test_noise)
    params = ['r', 'n']
    sap.predict_plot(y_test_noise, ypr, params, 'Предсказание индикатрис с шумом в 3%. \nОбучающая выборка с шумом 3%.')
    sap.error_plot(y_test_noise, ypr, params, 'Ошибка предсказания для индикатрис шумом в 3%. \nОбучающая выборка с шумом 3%.')

