import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import tensorflow as tf
import sklearn
#from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans

class RBFNet(object):

    def __init__(self, norm_type = 'L2', dist_type = 'std', quantile = 0.1):
        self.__c = None
        self.__d = None
        self.__wb = None
        self.norm_type = norm_type
        self.dist_type = dist_type
        self.quantile = quantile

    def norm(self, x1, x2):
        if self.norm_type == 'L2':
            return np.linalg.norm(x1 - x2)

        elif self.norm_type == 'Lihoswai':
            return np.sum(np.abs(x1 / x2 + x2 / x1 - 2))

        elif self.norm_type == 'L1':
            return np.linalg.norm(x1 - x2, ord=1)


    def fit(self, x_train, y_train, n_centers=None, centers=None):
        __x = np.copy(x_train)
        __y = np.copy(y_train).T

        if (centers) == None:
            from sklearn.cluster import KMeans
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.cluster import Birch

            print('Start kmeans clustering')
            #kmeans = KMeans(n_clusters=n_centers, random_state=0, max_iter=80, verbose=1)
            if n_centers < len(x_train):
                kmeans = MiniBatchKMeans(n_clusters=n_centers, max_iter=80, batch_size = 30000, verbose=1)
                kmeans.fit(__x)
                self.__c = kmeans.cluster_centers_
            else:
                self.__c = x_train

            print('Clustering complete')

        elif (centers) != None:
            self.__c = centers

        self.__d = self.__d_constr(self.__c)
        __F = self.__f_p_constr(__x)
        self.__wb = self.__train(__y, __F)

    def __d_constr(self, c):
        n = len(c)
        d = np.zeros(n)
        m = np.zeros(n)
        for i in tqdm(range(n), "Distant progress"):
            for j in range(n):
                m[j] = self.norm(c[i], c[j])

            if self.dist_type == 'mean':
                d[i] = np.mean(m)
            if self.dist_type == 'std':
                d[i] = np.std(m)
            if self.dist_type == 'quantile':
                d[i] = np.quantile(m, self.quantile)
            if self.dist_type == 'median':
                d[i] = np.median(m)
            if self.dist_type == 'max_min':
                d[i] = np.max(m) - np.min(m)
        return d

    def f_constr(self, x):
        s = len(x)
        n = len(self.__c)
        f = np.zeros((n, s))

        for i in tqdm(range(n), "F progress"):
            for j in range(s):
                f[i][j] = self.norm(self.__c[i], x[j])
                #f[i][j] = np.linalg.norm(x[j] - self.__c[i]) / np.linalg.norm(self.__c[i])
                #f[i][j] = np.linalg.norm(x[j] - self.__c[i])
                #f[i][j] =  np.sum(np.abs(x[j] / self.__c[i] + self.__c[i] / x[j] - 2))
            #f[i] = f[i] / self.__d[i]
        f = f / np.reshape(self.__d, (-1, 1))
        f = np.exp(-(f ** 2))
        return f

    def __f_p_constr(self, x):

        streams = 6
        splits = 6
        x_list = np.array_split(x, splits, axis=0)
        with mp.Pool(processes=streams) as pool:
            ff = pool.map(func=self.f_constr, iterable=x_list)
        pool.close()
        pool.join()
        pool.terminate()
        f = np.hstack(ff)
        return f

    def __fa_constr(self, f):
        a = np.ones((1, len(f[0])))
        return np.vstack((f, a))

    def __train(self, y, f):
        print('Start training')
        fa = self.__fa_constr(f)

        fafat = np.matmul(fa, fa.T)

        fafat = tf.linalg.pinv(fafat, rcond=10 ** (-100))
        wb = tf.linalg.matmul(y, fa, transpose_b=True)
        wb = tf.linalg.matmul(wb, fafat)
        wb = tf.make_tensor_proto(wb)
        wb = tf.make_ndarray(wb)
        print('Train complete')
        return wb

    def predict(self, x, multi=True):
        print('Start prediction')

        if multi == True:
            f = self.__f_p_constr(x)
        elif multi == False:
            f = self.f_constr(x)

        fa = self.__fa_constr(f)
        y = np.dot(self.__wb, fa)
        print('Prediction complete')
        return y.T

    def save_model(self, path):
        np.savetxt(path + r'\c.txt', self.__c, fmt= '%.18e')
        np.savetxt(path + r'\wb.txt', self.__wb, fmt='%.18e')
        np.savetxt(path + r'\d.txt', self.__d, fmt='%.18e')

    def load_model(self, path):
        self.__c = np.loadtxt(path + r'\c.txt')
        self.__wb = np.loadtxt(path + r'\wb.txt')
        self.__d = np.loadtxt(path + r'\d.txt')
    pass

class DRBF_Net(object):
    def __init__(self, N1=10, N2=150, dist_type='median', norm_type='L2'):
        from sklearn.cluster import MiniBatchKMeans
        self.RBF_list = []
        self.N1 = N1
        self.N2 = N2
        self.y_len = None
        self.x_len = None
        self.kmeans_n1 = MiniBatchKMeans(n_clusters=self.N1, max_iter=80, batch_size=30000, verbose=1)
        self.dist_type = dist_type
        self.norm_type = norm_type

    def fit(self, x_train, y_train):

        self.y_len = len(y_train[0])
        self.x_len = len(x_train[0])

        self.kmeans_n1.fit(x_train)
        labels = self.kmeans_n1.labels_

        for i in tqdm(range(self.N1)):
            x_buf = np.zeros((0, len(x_train[0])))
            y_buf = np.zeros((0, len(y_train[0])))
            for j in range(len(x_train)):
                if labels[j] == i:
                    x_buf = np.vstack((x_buf, x_train[j]))
                    y_buf = np.vstack((y_buf, y_train[j]))

            model = RBFNet(norm_type=self.norm_type, dist_type=self.dist_type)
            model.fit(x_buf, y_buf, n_centers=self.N2)
            self.RBF_list.append(model)
            del model

    def predict(self, x):
        y_predict = np.zeros((0, self.y_len))
        __labels = self.kmeans_n1.predict(x)

        for i in range(len(x)):
            _yb = self.RBF_list[__labels[i]].predict(np.reshape(x[i], (1,-1)), multi=False)
            y_predict = np.vstack((y_predict, _yb))
        return y_predict




