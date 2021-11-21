import numpy as np
from sklearn.cluster import KMeans


class PipeLine(object):
    def __init__(self):
        self.__x = None
        self.__y = None
        self.__x_edited = None
        self.__y_edited = None

        self.__x_forward = None
        self.__x_backward = None

        self.__x_forward_edited = None
        self.__x_backward_edited = None

        self.__angles_range = None
        self.__angles_array = None
        self.__angles_pi_array = None
        self.__c = None

    def load_data(self, x, y, angles_range):
        self.__x = x
        self.__y = y
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
        return self

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
        return self

    def cutter(self, angles_f,  angles_b):
        start_f = 0
        end_f = 0
        for i in range(len(self.__angles_array)):
            while self.__angles_array[i] <= angles_f[0]:
                start_f = i
                i = i + 1

            while self.__angles_array[i] <= angles_f[1]:
                end_f = i
                i = i + 1
        start_b = 0
        end_b = 0
        for i in range(len(self.__angles_array)):
            while self.__angles_array[i] <= angles_b[0]:
                start_b = i
                i = i + 1

            while self.__angles_array[i] <= angles_b[1]:
                end_b = i
                i = i + 1

        self.__x_forward = self.__x_edited[:, start_f:end_f + 1]
        self.__x_backward = self.__x_edited[:, start_b:end_b + 1]
        self.__x_forward_edited = np.copy(self.__x_forward)
        self.__x_backward_edited = np.copy(self.__x_backward)
        return self

    def reset(self):
        self.__x_edited = self.__x
        return self

    def noise_gen(self, noise_level_forward, noise_level_backward, replicas=0, noise_type='additive'):
        sigma_f = noise_level_forward * np.max(self.__x_forward)
        sigma_b = noise_level_backward * np.max(self.__x_backward)
        self.__y_edited = self.__y
        if replicas > 0:
            for k in range(replicas):
                x_f_buf = np.copy(self.__x_forward)
                x_b_buf = np.copy(self.__x_backward)
                for i in range(len(x_f_buf)):
                    if noise_type == 'multiplicative':
                        sigma_f = np.max((x_f_buf[i])) * noise_level_forward
                        sigma_b = np.max((x_b_buf[i])) * noise_level_backward
                    x_f_buf[i] = np.random.normal(x_f_buf[i], sigma_f)
                    x_b_buf[i] = np.random.normal(x_b_buf[i], sigma_b)

                self.__x_forward_edited = np.vstack([self.__x_forward_edited, x_f_buf])
                self.__x_backward_edited = np.vstack([self.__x_backward_edited, x_b_buf])
                self.__y_edited = np.vstack([self.__y_edited, self.__y])

        else:
            x_f_buf = np.copy(self.__x_forward)
            x_b_buf = np.copy(self.__x_backward)
            for i in range(len(x_f_buf)):
                if noise_type == 'multiplicative':
                    sigma_f = np.max((x_f_buf[i])) * noise_level_forward
                    sigma_b = np.max((x_b_buf[i])) * noise_level_backward
                x_f_buf[i] = np.random.normal(x_f_buf[i], sigma_f)
                x_b_buf[i] = np.random.normal(x_b_buf[i], sigma_b)

            self.__x_forward_edited = x_f_buf
            self.__x_backward_edited = x_b_buf
        return self

    def get_data(self):
        return [self.__x_forward_edited, self.__x_backward_edited,  self.__y_edited]

    def kmean_clustrer(self, n):
        print('Start kmeans clustering')
        kmeans = KMeans(n_clusters=n, random_state=0, max_iter=40, verbose=1)
        kmeans.fit(self.__x_forward)
        print('Clustering complete')
        return kmeans.cluster_centers_


class Opti(object):
    def __init__(self):
        self.__bd_forward = None
        self.__bd_backward = None
        self.__params = None

    def load_data(self, monomer):
        self.__bd_forward = monomer[0]
        self.__bd_backward = monomer[1]
        self.__params = monomer[2]

    def fit(self, x):
        ll = np.linalg.norm(self.__bd_forward - x*np.ones(np.shape(self.__bd_forward)), axis=1)
        # ll = np.linalg.norm(self.__bd_forward / (x * np.ones(np.shape(self.__bd_forward))) - 1, axis=1)
        optima_id = np.argmin(ll)
        return [self.__bd_forward[optima_id], self.__bd_backward[optima_id], self.__params[optima_id]]
