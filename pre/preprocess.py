import os
import os.path
import scipy.io as sio
import numpy as np
import struct
import random
from pca import PCA
import pcl
from farthest_point_sampling_fast import FPSF

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

    def main(self):

        for sub_idx in range(len(sub_names)):

            os.mkdir(os.path.join(save_dir, sub_names[sub_idx]))

            for ges_idx in range(len((ges_names))):

                ges_dir = os.path.join(
                    dataset_dir, sub_names[sub_idx], ges_names[ges_idx])
                [framr_num, gt_wld] = self.read_ground_truth(ges_dir)
                save_ges_dir = os.path.join(
                    save_dir, sub_names[sub_idx], ges_names[ges_idx])
                os.mkdir(save_ges_dir)
                [valid,hand_points,jnt_xyz] = self.read_conv_depth(ges_dir, sub_idx, ges_idx,gt_wld)
                print(save_ges_dir)
                

    def read_ground_truth(self, ges_dir):
        # 1. read ground truth
        with open(ges_dir + '/joint.txt', 'r') as f:
            frame_num = int(f.readline(1))
            # 跳过第一行，不然会显示numpy 列对不上，第一行只有 frame_num 一个数值，比下面的少
            A = np.loadtxt(ges_dir, skiprows=1)
            gt_wld = A.reshape(3, 21, frame_num)
            gt_wld[3, :, :] = -gt_wld[3]
            # 此处有两种方法，还一种为 self.gt_wld.transpose(2,1,0)
            gt_wld = gt_wld.swapaxis(2, 0)

        return frame_num, gt_wld

    def read_conv_depth(self, ges_dir, sub_idx, ges_idx, gt_wld):

        num = 0
        for i in os.listdir(ges_dir):
            if os.path.splitext(i)[1] == '.bin':
                num += 1
        valid = self.msra_valid[sub_idx, ges_idx]
        for frm_idx in range(num):
            if not valid(frm_idx):
                continue
        # read binary files
            with open(ges_dir + '/' + str('%06d' % frm_idx) + '_depth.bin', 'r') as f:
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
                        hand_3d[idx, 1] = -(img_width/2 - (jj + bb_left-1)
                                            ) * hand_depth(ii, jj)/fFocal_msra
                        hand_3d[idx, 2] = -(img_height/2 - (ii + bb_top-1)
                                            ) * hand_depth(ii, jj) / fFocal_msra
                        hand_3d[idx, 3] = hand_depth(ii, jj)

                valid_idx = []

                for num in range(valid_pixel_num):
                    if any(hand_3d[num, :]):
                        valid_idx.append(num)

                hand_points = hand_3d[valid_idx, :]
                jnt_xyz = np.squeeze(gt_wld[frm_idx, :, :])

                [coeff, pt_Cloud, hand_points_rotate]=self.OOB_PCA(hand_points)
                [pc, max_bb3d_len, offset,jnt_xyz_normalized]=self.sampling_nomalizing(hand_points,hand_points_rotate,coeff,pt_Cloud,jnt_xyz)

        return valid,hand_points,jnt_xyz

    def OOB_PCA(self, hand_points):
        coeff = PCA.pca(hand_points)[0]
        if coeff[2, 1] < 0:
            coeff[:, 1] = -coeff[:, 1]
        if coeff[3, 3] < 0:
            coeff[:, 3] = -coeff[:, 3]
        coeff[:, 2] = np.dot(coeff[:, 3], coeff[:, 1])

        # 需要修改
        # ptCloud = pointCloud(hand_points);
        pt_cloud = pcl.PointCloud(hand_points)
        hand_points_rotate = hand_points*coeff

        return coeff, pt_cloud, hand_points_rotate

    def sampling_nomalizing(self, hand_points,hand_points_rotate,coeff,ptCloud,jnt_xyz):
        hand_shape = hand_points.shape[0]
        if hand_shape < self.sample_num:
            tmp = np.floor(self.sample_num / hand_shape)
            rand_ind = []
            for tmp_i in range(tmp):
                rand_ind += [i for i in range(hand_shape)]

            rand_ind += np.random.randint(0, hand_shape,
                                          size=self.sample_num % hand_shape)
        else:
            rand_ind = np.random.randint(
                0, hand_shape, size=self.sample_num % hand_shape)
        hand_points_sampled = hand_points[rand_ind, :]
        hand_points_rotate_sampled = hand_points_rotate[rand_ind, :]    
        
        normal_k = 30
        # 需要修改
        # normals = pcnormals(ptCloud, normal_k);
        normals = pcl.PointNormal(ptCloud, normal_k)
        normals_sampled = normals[rand_ind, :]

        sensor_center = [0, 0, 0]
        for k in range(self.sample_num):
            p1 = sensor_center - hand_points_sampled[k, :]

            # 可能需要修改
            # angle = atan2(norm(cross(p1,normals_sampled(k,:))),p1*normals_sampled(k,:)');
            angle = np.arctan2(np.linalg.norm(
                p1*normals_sampled[k, :]), p1*normals_sampled[k, :].transpose())
            if angle > np.pi / 2 or angle < -np.pi / 2:
                normals_sampled[k, :] = -normals_sampled[k, :]

            normals_sampled_rotate = normals_sampled * coeff

            # Normalize point cloud
            x_min_max = [min(hand_points_rotate[:, 1],
                             max(hand_points_rotate[:, 1]))]
            y_min_max = [min(hand_points_rotate[:, 2],
                             max(hand_points_rotate[:, 2]))]
            z_min_max = [min(hand_points_rotate[:, 3],
                             max(hand_points_rotate[:, 3]))]

            scale = 1.2
            bb3d_x_len = scale * (x_min_max[1] - x_min_max[0])
            bb3d_y_len = scale * (x_min_max[1] - x_min_max[0])
            bb3d_z_len = scale * (x_min_max[1] - x_min_max[0])
            max_bb3d_len = bb3d_x_len

            hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
            if hand_shape < self.sample_num:
                offset = np.mean(hand_points_rotate) / max_bb3d_len
            else:
                offset = np.mean(hand_points_normalized_sampled)

            hand_points_normalized_sampled -= offset

            pc = [hand_points_normalized_sampled, normals_sampled_rotate]

            # 需要修改
            # sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1)';
            sampled_idx_l1 = FPSF(hand_points_normalized_sampled, self.sample_num_level1).fpsf().transpose()
            other_idx = np.setdiff1d(
                [i for i in self.sample_num], sampled_idx_l1)
            new_idx = [sampled_idx_l1, other_idx]
            # pc = pc[new_idx, :]
            sort_idx = np.sort(new_idx)
            for i in sort_idx:
                pc[i, :] = pc[new_idx[i], :]
                # 需要修改
                # sampled_idx_l2 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level2)';
            sampled_idx_l2 = FPSF(hand_points_normalized_sampled, self.sample_num_level2).fpsf().transpose()
            other_idx = np.setdiff1d(
                [i for i in self.sample_num], sampled_idx_l2)
            new_idx = [sampled_idx_l2, other_idx]
            # pc[i for i in self.sample_num, :] = pc[new_idx, :]
            for i in range(self.sample_num):
                pc[i, :] = pc[new_idx[i], :]

            # ground truth
            jnt_xyz_normalized = (jnt_xyz * coeff) / max_bb3d_len
            jnt_xyz_normalized -= offset

        return pc, max_bb3d_len, offset,jnt_xyz_normalized

    def save_mat(self, frame_num,frm_idx, coeff, save_ges_dir,value,valid):
        # initial point cloud and surface normal
        Point_Cloud_FPS = np.zeros((frame_num, self.sample_num, 6))
        Volume_rotate = np.zeros((frame_num, 3, 3))
        Volume_lenth = np.zeros((frame_num, 1))
        Volume_offset = np.zeros((frame_num, 3))
        Volume_GT_XYZ = np.zeros((frame_num, self.joint_num, 3))

        # get value
        [pc, max_bb3d_len, offset,jnt_xyz_normalized]=value
        Point_Cloud_FPS[frm_idx, :, :] = pc
        Volume_rotate[frm_idx, :, :] = coeff
        Volume_lenth[frm_idx] = max_bb3d_len
        Volume_offset[frm_idx, :] = offset
        Volume_GT_XYZ[frm_idx, :, :] = jnt_xyz_normalized

        # save
        sio.savemat(save_ges_dir + '/Point_Cloud_FPS.mat', Point_Cloud_FPS)
        sio.savemat(save_ges_dir + '/Volume_rotate.mat', Volume_rotate)
        sio.savemat(save_ges_dir + '/Volume_length.mat', Volume_lenth)
        sio.savemat(save_ges_dir + '/Volume_offset.mat', Volume_offset)
        sio.savemat(save_ges_dir + '/Volume_GT_XYZ.mat', Volume_GT_XYZ)
        sio.savemat(save_ges_dir+'/valid.mat', valid)
