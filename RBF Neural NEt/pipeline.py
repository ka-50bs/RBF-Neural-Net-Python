import numpy as np

def weighting(x, func_type, angl):
    n = len(x[0])
    k = (angl[1] - angl[0]) / (n - 1)

    angles_pi = np.zeros(n)
    angles = np.zeros(n)

    for i in range(n):
        angles_pi[i] = (angl[0] + k * i) * np.pi / 180
        angles[i] = angl[0] + k * i

    if func_type == 'sin':
        weight = np.sin(angles_pi[i])
        return x * weight

    if func_type == 'sin^2':
        weight = np.sin(angles_pi[i]) ** 2
        return x * weight

    if func_type == 'sin^4':
        weight = np.sin(angles_pi[i]) ** 4
        return x * weight

    if func_type == 'log':
        return np.log(x)

    if func_type == 'm_func':
        weight = np.copy(angles)
        for i in range(len(angles)):
            if (angles[i] > 0) and (angles[i] < 90):
                weight[i] = (np.exp(-2 * (np.log(angles[i] / 54.0)) ** 2)) / angles[i]
            elif (angles[i] > 90) and (angles[i] < 180):
                weight[i] = (np.exp(-2 * (np.log((180 - angles[i]) / 54.0)) ** 2)) / (180 - angles[i])
            else:
                weight[i] = 0
        return x * weight


def cutter(x, start, end):
    return x[:, start:end + 1]

