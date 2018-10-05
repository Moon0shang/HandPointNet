import os
import os.path
import scipy.io as sio
import numpy as np
import struct

dataset_dir = "D:/Cache/Git/HandPointNet/data/cvpr15_MSRAHandGestureDB/"
save_dir = "./"

sub_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
ges_names = ['1', '2', '3', '4', '5', '6', '7', '8',
             '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']


class preprocess(object):

    def __init__(self, joint_num=21, sample_num=1024, sample_num_level1=512, sample_num_level2=128):

        self.joint_num = joint_num
        self.sample_num = sample_num
        self.sample_num_level1 = sample_num_level1
        self.sample_num_level2 = sample_num_level2
        self.msra_valid = sio.loadmat("./msra_valid.mat")
        self.initail_point_cloud()

    def main(self):

        for sub_idx in range(len(sub_names)):

            os.mkdir(os.path.join(save_dir, sub_names[sub_idx]))

            for ges_idx in range(len((ges_names))):

                ges_dir = os.path.join(
                    dataset_dir, sub_names[sub_idx], ges_names[ges_idx])
                self.read_ground_truth(ges_dir)
                save_ges_dir = os.path.join(
                    save_dir, sub_names[sub_idx], ges_names[ges_names])
                os.mkdir(save_ges_dir)
                print(save_ges_dir)
                self.read_depth_files(path, sub_idx, ges_idx)

        return ges_dir

    def read_ground_truth(self, path):
        # 1. read ground truth
        with open(path + '/joint.txt', 'r') as f:
            self.frame_num = int(f.readline(1))
            # 跳过第一行，不然会显示numpy 列对不上，第一行只有 frame_num 一个数值，比下面的少
            A = np.loadtxt(path, skiprows=1)
            self.gt_wld = A.reshape(3, 21, self.frame_num)
            self.gt_wld[3, :, :] = -self.gt_wld[3]
            # 此处有两种方法，还一种为 self.gt_wld.transpose(2,1,0)
            self.gt_wld = self.gt_wld.swapaxis(2, 0)

    def __initail_point_cloud(self):
        # initial point cloud and surface normal
        point_cloud_FPS = np.zeros((self.frame_num, self.sample_num, 6))
        volume_rotate = np.zeros((self.frame_num, 3, 3))
        volume_length = np.zeros((self.frame_num, 1))
        volume_offset = np.zeros((self.frame_num, 3))
        volume_GT_XYZ = np.zeros((self.frame_num, self.joint_num, 3))

    def read_conv_depth(self, path，sub_idx, ges_idx):
        num = 0
        for i in os.listdir(path):
            if os.path.splitext(i)[1] = '.bin':
                num += 1
        valid = self.msra_valid[sub_idx, ges_idx]
        for frm_idx in range(num):
            if not valid(frm_idx):
                continue
        # read binary files
            with open(path + '/' + str('%06d' % frm_idx) + '_depth.bin', 'r') as f:
                img_width = struct.unpack('I', f.read(4))[0]
                img_height = struct.unpack('I', f.read(4))[0]

                bb_left = struct.unpack('I', f.read(4))[0]
                bb_top = struct.unpack('I', f.read(4))[0]
                bb_right = struct.unpack('I', f.read(4))[0]
                bb_bottom = struct.unpack('I', f.read(4))[0]
                bb_width = bb_right - bb_left
                bb_height = bb_bottom - bb_top

                valid_pixel_num = bb_width * bb_height

                hand_depth = struct.unpack(
                    'f' * valid_pixel_num, f.read(valid_pixel_num))
                hand_depth = np.array(hand_depth, dtype=np.float32)
                hand_depth = hand_depth.reshape(bb_width, bb_height)
                hand_depth = hand_depth.transpose()

                # convert depth to xyz
                fFocal_msra = 241.42
                hand_3d = np.zeros((valid_pixel_num, 3))
                for ii in bb_height:
                    for jj in bb_width:
                        idx = jj * bb_height + ii+1
                        hand_3d[idx, 1] = -(img_width/2 - (jj + bb_left-1))*hand_depth(ii,jj)/fFocal_msra
                       
