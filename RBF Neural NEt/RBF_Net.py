import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import multiprocessing as mp

class RBFNet(object):

    def __init__(self):
        return

    def fit(self, x_train, y_train, n):
        __x = np.copy(x_train)
        __y = np.copy(y_train).T

        self.__c = self.__k_clustrer(__x, n)
        self.__d = self.__d_constr(self.__c)

        #__F = self.__f_constr(__x, self.__c, self.__d)
        __F = self.__f_p_constr(__x)

        self.__wb = self.__train(__y, __F)

    def __k_clustrer(self, x, n):
        kmeans = KMeans(n_clusters=n, random_state=0, max_iter=40, verbose=1)
        kmeans.fit(x)
        return kmeans.cluster_centers_

    def __d_constr(self, c):
        n = len(c)
        d = np.zeros(n)
        m = np.zeros(n)
        for i in tqdm(range(n), "Distant progress"):
            for j in range(n):
                m[j] = np.linalg.norm(c[i] - c[j])
            d[i] = np.std(m)
            #d[i] = np.median(m)
        return d

    def f_constr(self, x):
        s = len(x)
        n = len(self.__c)
        f = np.zeros((n, s))

        for i in tqdm(range(n), "F progress"):
            for j in range(s):
                f[i][j] = np.linalg.norm(x[j] - self.__c[i])
            f[i] = f[i] / self.__d[i]
        f = np.exp(-(f ** 2))
        return f

    def __f_p_constr(self, x):
        streams = 12
        x_list = np.array_split(x, streams, axis=0)
        pool = mp.Pool(streams)
        f = np.hstack(pool.map(self.f_constr, x_list))
        pool.close()
        pool.join()
        return f

    def __fa_constr(self, f):
        a = np.ones((1, len(f[0])))
        return np.vstack((f, a))

    def __train(self, y, f):
        import tensorflow as tf
        fa = self.__fa_constr(f)
        fafat = tf.linalg.matmul(fa, fa, transpose_b=True)
        fafat = tf.linalg.pinv(fafat, rcond=10 ** (-100), validate_args=True)
        wb = tf.linalg.matmul(y, fa, transpose_b=True)
        wb = tf.linalg.matmul(wb, fafat)
        wb = tf.make_tensor_proto(wb)
        wb = tf.make_ndarray(wb)
        print('Train complete')

        return wb

    def predict(self, x):
        f = self.__f_p_constr(x)
        fa = self.__fa_constr(f)
        y = np.dot(self.__wb, fa)
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
