'''
load hand point data
'''
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
# import pdb

SAMPLE_NUM = 1024
JOINT_NUM = 21

subject_names = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
gesture_names = ["1", "2", "3", "4", "5", "6", "7", "8",
                 "9", "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"]


class HandPointDataset(data.Dataset):
    def __init__(self, root_path, opt, train=True):
        self.root_path = root_path
        self.train = train
        self.size = opt.size    # load多少样本，可选 full / samll，默认为 full
        self.test_index = opt.test_index    # 交叉验证试验指标 , 数据文件：0~8

        self.PCA_SZ = opt.PCA_SZ    # PCA 成分数量大小，默认为 42(int)
        self.SAMPLE_NUM = opt.SAMPLE_NUM  # 样本点数量，默认为 1024(int)
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # 输入点的特征值， 默认为 6(int)
        self.JOINT_NUM = opt.JOINT_NUM  # joint 数量， 默认为 21(int)

        if self.size == 'full':
            self.SUBJECT_NUM = 9    # 一共有9组预处理完的数据
            self.GESTURE_NUM = 17   # 每组预处理数据下有17个小分类
        elif self.size == 'small':
            self.SUBJECT_NUM = 3
            self.GESTURE_NUM = 2

        self.total_frame_num = self.__total_frmae_num()  # 从Volume Length中获取的长度

        # 初始化 python 数组

        # 3维矩阵， 包含 toal_frame_num 个 sample_num  行 Input_num 列的二维矩阵
        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        # 2维矩阵 ，toal_frame_num 行 1列
        self.volume_length = np.empty(
            shape=[self.total_frame_num, 1], dtype=np.float32)
        # 3维矩阵 ，包含 toal_frame_num 个 joint_num 行 3 列的二维矩阵
        self.gt_xyz = np.empty(
            shape=[self.total_frame_num, self.JOINT_NUM, 3], dtype=np.float32)
        # 2维矩阵， toal_frame_num 行 1 列
        self.valid = np.empty(
            shape=[self.total_frame_num, 1], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0
        # 获取对应的训练数据的路径并加载数据
        # 选去一个文件P0的数据作为测试数据，其他数据维训练数据
        # 将所有的预处理数据合并到一个变量中去（训练/测试数据分开，需要调用这个模块2次）
        if self.train:  # train
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject != self.test_index:
                    for i_gesture in range(self.GESTURE_NUM):
                        cur_data_dir = os.path.join(
                            self.root_path, subject_names[i_subject], gesture_names[i_gesture])
                        print("Training: " + cur_data_dir)
                        self.__loaddata(cur_data_dir)
        else:  # test
            for i_gesture in range(self.GESTURE_NUM):
                cur_data_dir = os.path.join(
                    self.root_path, subject_names[self.test_index], gesture_names[i_gesture])
                print("Testing: " + cur_data_dir)
                self.__loaddata(cur_data_dir)

        self.point_clouds = torch.from_numpy(self.point_clouds)
        self.volume_length = torch.from_numpy(self.volume_length)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)
        self.valid = torch.from_numpy(self.valid)

        # 转换矩阵的size为 total_frame_num 行的二维矩阵
        self.gt_xyz = self.gt_xyz.view(self.total_frame_num, -1)
        valid_ind = torch.nonzero(self.valid)   # 返回valid 中非 0 数据的索引
        valid_ind = valid_ind.select(1, 0)  # 选取第1 行的第0 个

        # 切片：index_select(dim, range(必须为tensor.long)); dim= 0 行切片
        # 将 valid_int 转换为long 类型数据
        self.point_clouds = self.point_clouds.index_select(0, valid_ind.long())
        self.volume_length = self.volume_length.index_select(
            0, valid_ind.long())
        self.gt_xyz = self.gt_xyz.index_select(0, valid_ind.long())
        self.total_frame_num = self.point_clouds.size(0)

        # load PCA coeff
        PCA_data_path = os.path.join(
            self.root_path, subject_names[self.test_index])
        print("PCA_data_path: " + PCA_data_path)
        PCA_coeff_mat = sio.loadmat(
            os.path.join(PCA_data_path, 'PCA_coeff.mat'))
        # 截取前 PCA_SZ 列数据
        self.PCA_coeff = torch.from_numpy(
            PCA_coeff_mat['PCA_coeff'][:, 0:self.PCA_SZ].astype(np.float32))
        PCA_mean_mat = sio.loadmat(os.path.join(
            PCA_data_path, 'PCA_mean_xyz.mat'))
        self.PCA_mean = torch.from_numpy(
            PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))
        # 将一列的值扩展至 joint_num 列
        tmp = self.PCA_mean.expand(self.total_frame_num, self.JOINT_NUM * 3)
        tmp_demean = self.gt_xyz - tmp
        # 矩阵相乘
        self.gt_pca = torch.mm(tmp_demean, self.PCA_coeff)
        # 转置 transpose(dim1, dim2)
        self.PCA_coeff = self.PCA_coeff.transpose(0, 1).cuda()
        self.PCA_mean = self.PCA_mean.cuda()

    def __getitem__(self, index):
        '''
        获取指定index 的数值
        '''
        return self.point_clouds[index, :, :], self.volume_length[index], self.gt_pca[index, :], self.gt_xyz[index, :]

    def __len__(self):
        return self.point_clouds.size(0)

    def __loaddata(self, data_dir):
        point_cloud = sio.loadmat(os.path.join(
            data_dir, 'Point_Cloud_FPS.mat'))
        gt_data = sio.loadmat(os.path.join(data_dir, "Volume_GT_XYZ.mat"))
        volume_length = sio.loadmat(
            os.path.join(data_dir, "Volume_length.mat"))
        valid = sio.loadmat(os.path.join(data_dir, "valid.mat"))

        self.start_index = self.end_index + 1
        self.end_index = self.end_index + len(point_cloud['Point_Cloud_FPS'])

        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud['Point_Cloud_FPS'].astype(
            np.float32)
        self.gt_xyz[(self.start_index - 1):self.end_index, :,
                    :] = gt_data['Volume_GT_XYZ'].astype(np.float32)
        self.volume_length[(self.start_index - 1):self.end_index,
                           :] = volume_length['Volume_length'].astype(np.float32)
        self.valid[(self.start_index - 1):self.end_index,
                   :] = valid['valid'].astype(np.float32)

    def __total_frmae_num(self):
        frame_num = 0
        if self.train:  # train
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject != self.test_index:
                    for i_gesture in range(self.GESTURE_NUM):
                        cur_data_dir = os.path.join(
                            self.root_path, subject_names[i_subject], gesture_names[i_gesture])
                        frame_num = frame_num + \
                            self.__get_frmae_num(cur_data_dir)
        else:  # test
            for i_gesture in range(self.GESTURE_NUM):
                cur_data_dir = os.path.join(
                    self.root_path, subject_names[self.test_index], gesture_names[i_gesture])
                frame_num = frame_num + self.__get_frmae_num(cur_data_dir)
        return frame_num

    def __get_frmae_num(self, data_dir):
        volume_length = sio.loadmat(
            os.path.join(data_dir, "Volume_length.mat"))
        return len(volume_length['Volume_length'])
