'''
farthest point sampling
point_cloud: Nx3
'''

import numpy as np


class FPSF(object):
    def __init__(self, point_cloud, sample_num):
        self.point_cloud = point_cloud
        self.sample_num = sample_num
        self.pc_num = point_cloud.shape[0]

    def fpsf(self):

        if self.pc_num <= self.sample_num:
            sampled_idx = np.array(
                range(self.pc_num), dtype=np.int32).transpose()
            rand_m = np.random.randint(
                1, self.pc_num, size=(self.sample_num-self.pc_num, 1))
            sampled_idx = np.append(sampled_idx, rand_m, axis=0)
        else:
            sampled_idx = np.zeros(self.sample_num, 1)
            sampled_idx[1] = np.random.randint(1, self.sample_num)

            diff_n = self.point_cloud - sampled_idx
            min_dist = np.sum(np.dot(diff_n, diff_n), axis=1)

            # 可能需要修改
            for i in [i for i in range(self.sample_num)].pop(0):
                sampled_idx[i] = np.max(min_dist)
                if i < self.sample_num:
                    valid_idx = [min_dist > np.exp(-8)]
                    diff_n = self.point_cloud(
                        valid_idx) - self.point_cloud(sampled_idx)
                    for i in valid_idx:
                        min_dist[valid_idx[i], :] = np.min(
                            valid_idx, axis=np.sum(np.dot(diff_n, diff_n), axis=1))

        sampled_idx = np.unique(sampled_idx)

        return sampled_idx
