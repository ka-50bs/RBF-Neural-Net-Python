import matplotlib.pyplot as plt
import numpy as np

def predict_plot(y_test, y_pred, params):
    N = len(params)

    plt.figure(figsize=(20, 20))

    for i in range(N):
        for j in range(N):
            if i == j:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Предсказание с помощью сетей")
                plt.xlabel(params[i] + ' Тестовые')
                plt.ylabel(params[i] + ' Предсказанные')

                plt.plot(y_test[:, i], y_pred[:, i], '*', label="Тест")

                plt.legend()
            else:
                plt.subplot(N, N, i * N + j + 1)
                plt.title("Предсказание с помощью сетей")
                plt.xlabel(params[i])
                plt.ylabel(params[j])

                plt.plot(y_test[:, i], y_test[:, j], '*', label="Тест")
                plt.plot(y_pred[:, i], y_pred[:, j], '.', label="Предсказание")
                plt.legend()
    plt.show()


def error_plot(y_test, y_pred, params):
    N = len(params)

    error = []
    for j in range(len(y_pred[0])):
        buf_err = []
        for i in range(len(y_pred)):
            buf_err.append((y_test[i, j] - y_pred[i, j]))
        error.append(buf_err)

    plt.figure(figsize=(20, 20))

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
        plt.text(0, 0, 'mu = %f \n sigma = %f' % (np.mean(error[i]), np.std(error[i])), bbox=dict(facecolor='white'))
    plt.show()
    return error
