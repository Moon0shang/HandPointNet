'''
PCA
'''

import numpy as np


class PCA(object):

    def __init__(self, data):
        self.data = data
        self.col = data.shape[1]
        # self.ouput_var()

    # first step: normalize the matrix
    def cal_mean(self):
        mean_value = np.mean(self.data)
        # 要想正确使用std， 必须设置参数ddof=1
        # 数据量较大时结果误差可忽略，但数据量小时必须注意
        std_value = np.std(self.data, ddof=1)
        self.data = (self.data - mean_value) / std_value

    # second: calculate coefficients, feature vector and feature value
    def cal_features(self):
        Relate_coef_func = np.corrcoef(self.data)
        f_value, f_vector = np.linalg.eig(Relate_coef_func)
        coeff = np.rot90(f_vector).transpose()
        f_value = np.rot90(np.rot90(f_value))
        latent = np.diag(f_value)[:, np.newaxis]

        return coeff, latent

    # third: calculate ratio
    def cal_ratio(self, latent):
        ratio = 0
        for i in range(len(self.col)):
            r = latent[i] / np.sum(latent)
            ratio += r
            if ratio >= 0.95:
                break

    # ouput values
    def pca(self):

        self.cal_mean()
        coeff, latent = self.cal_features()
        self.cal_ratio(latent)
        score = coeff * self.data

        return coeff, latent, score
