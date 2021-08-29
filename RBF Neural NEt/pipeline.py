import numpy as np


class PipeLine(object):
    def __init__(self):
        self.__x = None
        self.__y = None
        self.__x_edited = None
        self.__y_edited = None
        self.__angles_range = None
        self.__angles_array = None
        self.__angles_pi_array = None

    def load_data(self, x, y, angles_range):
        self.__x = x
        self.__angles_range = angles_range
        self.__x_edited = x
        self.__y_edited = y

        n = len(self.__x[0])
        k = (self.__angles_range[1] - self.__angles_range[0]) / (n - 1)

        angles_pi = np.zeros(n)
        angles = np.zeros(n)

        for i in range(n):
            angles_pi[i] = (self.__angles_range[0] + k * i) * np.pi / 180
            angles[i] = self.__angles_range[0] + k * i

        self.__angles_array = angles
        self.__angles_pi_array = angles_pi

    def weighting(self, func_type):
        angles_pi = self.__angles_pi_array
        angles = self.__angles_array

        if func_type == 'sin':
            weight = np.sin(angles_pi)
            self.__x_edited = self.__x_edited * weight
        elif func_type == 'sin^2':
            weight = np.sin(angles_pi) ** 2
            self.__x_edited = self.__x_edited * weight
        elif func_type == 'sin^4':
            weight = np.sin(angles_pi) ** 4
            self.__x_edited = self.__x_edited * weight
        elif func_type == 'm_func':
            weight = np.copy(angles)
            for i in range(len(angles)):
                if (angles[i] > 0) and (angles[i] < 90):
                    weight[i] = (np.exp(-2 * (np.log(angles[i] / 54.0)) ** 2)) / angles[i]
                elif (angles[i] > 90) and (angles[i] < 180):
                    weight[i] = (np.exp(-2 * (np.log((180 - angles[i]) / 54.0)) ** 2)) / (180 - angles[i])
                else:
                    weight[i] = 0
            self.__x_edited = self.__x_edited * weight
        elif func_type == 'log':
            self.__x_edited = np.log(self.__x_edited)

    def cutter(self, angles_diap):
        start = 0
        end = 0
        for i in range(len(self.__angles_array)):
            while self.__angles_array[i] <= angles_diap[0]:
                start = i
                i = i + 1

            while self.__angles_array[i] <= angles_diap[1]:
                end = i
                i = i + 1

        self.__x_edited = self.__x_edited[:, start:end + 1]

    def reset(self):
        self.__x_edited = self.__x

    def noise_generation(self, noise_level, replicas=0):
        if replicas > 0:
            for k in range(replicas):
                x_train_buf = np.copy(self.__x_edited)
                for i in range(len(x_train_buf)):
                    sigma = np.max((x_train_buf[i])) * noise_level
                    for j in range(len(x_train_buf[i])):
                        x_train_buf[i][j] = np.random.normal(x_train_buf[i][j], sigma)
                self.__x_edited = np.vstack([self.__x_edited, x_train_buf])
                self.__y_edited = np.vstack([self.__y_edited, self.__y_edited])
        else:
            x_train_buf = np.copy(self.__x_edited)
            for i in range(len(x_train_buf)):
                sigma = np.max((x_train_buf[i])) * noise_level
                for j in range(len(x_train_buf[i])):
                    x_train_buf[i][j] = np.random.normal(x_train_buf[i][j], sigma)
            self.__x_edited = x_train_buf

    def get_data(self):
        return [self.__x_edited, self.__y_edited]
