import matplotlib.pyplot as plt
import numpy as np


def predict_regr_plot(y_test, y_pred, params, sup_title):
    N = len(params)

    plt.figure(figsize=(40, 40))
    #plt.suptitle(sup_title)
    for i in range(N):
        for j in range(N):
            if i == j:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Корреляционный график")
                plt.xlabel(params[i] + ' Тестовые')
                plt.ylabel(params[i] + ' Предсказанные')

                plt.plot(y_test[:, i], y_pred[:, i], '*')

                #plt.legend()
            else:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Сравнительная карта параметров")
                plt.xlabel(params[i])
                plt.ylabel(params[j])

                plt.plot(y_test[:, i], y_test[:, j], '*', label="Тест")
                plt.plot(y_pred[:, i], y_pred[:, j], '.', label="Предсказание")
                plt.legend()
    plt.show()


def error_regr_plot(y_test, y_pred, params, sup_title):
    N = len(params)

    error = []
    for j in range(len(y_pred[0])):
        buf_err = []
        for i in range(len(y_pred)):
            buf_err.append((y_test[i, j] - y_pred[i, j]))
        error.append(buf_err)

    error = np.array(error)
    plt.figure(figsize=(200, 200))
    #plt.suptitle(sup_title)

    for i in range(N):
        plt.subplot(N, 2, i * 2 + 1)
        plt.title("Ошибка предсказания %s" % (params[i]))
        plt.ylabel("Ошибка предсказания %s" % (params[i]))
        plt.xlabel("Номер индикатрисы")
        plt.plot(error[i], '*')

        plt.subplot(N, 2, i * 2 + 2)
        plt.title("Распределение ошибки %s" % (params[i]))
        plt.xlabel("Ошибка предсказания %s" % (params[i]))
        plt.hist(error[i], 100)
        plt.text(x = 0, y = 0, s = 'RMSE = %f \n MAE = %f' % (
                                                                                   np.sqrt(np.mean(error[i] ** 2)),
                                                                                   np.mean(np.abs(error[i]))),
                 bbox=dict(facecolor='white'))
    plt.show()
    return error

def predict_map_plot(y_mon, y_dim, params, sup_title):
    N = len(params)

    plt.figure(figsize=(80, 80))
    plt.suptitle(sup_title)
    for i in range(N):
        for j in range(N):
            if i == j:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Корреляционный график")
                plt.xlabel(params[i] + 'Мономеры')
                plt.ylabel(params[i] + 'Димеры')

                plt.plot(y_mon[:, i], y_dim[:, i], '*', label="-")

                plt.legend()
            else:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Сравнительная карта параметров")
                plt.xlabel(params[i])
                plt.ylabel(params[j])

                plt.plot(y_mon[:, i], y_mon[:, j], '*', label="Мономеры")
                plt.plot(y_dim[:, i], y_dim[:, j], '.', label="Димеры")
                plt.legend()
    plt.savefig('fig.png', dpi=300)
    plt.show()

