import os
import os.path
import scipy.io as sio
import numpy as np

dataset_dir = "D:/Cache/Git/HandPointNet/data/cvpr15_MSRAHandGestureDB/"
save_dir = "./"

sub_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
ges_names = ['1', '2', '3', '4', '5', '6', '7', '8',
             '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']

joint_num = 21
sample_num = 1024
sample_num_level1 = 512
sample_num_level2 = 128

msra_valid = sio.loadmat("./msra_valid.mat")

for sub_idx in range(len(sub_names)):
    os.mkdir(os.path.join(save_dir, sub_names[sub_idx]))

    for ges_idx in range(len((ges_names))):
        ges_dir = os.path.join(
            dataset_dir, sub_names[sub_idx], ges_names[ges_idx])
        # depth files
        depth_files
        # 1. read ground truth
        with open(ges_dir + '/joint.txt', 'r') as f:
            frame_num = int(f.readline(1))
            # 跳过第一行，不然会显示numpy 列对不上，第一行只有 frame_num 一个数值，比下面的少
            A = np.loadtxt(ges_dir, skiprows=1)
            gt_wld = A.reshape(3, 21, frame_num)
            gt_wld[3, :, :] = -gt_wld[3]
            # 此处有两种方法，还一种为 gt_wld.transpose(2,1,0)
            gt_wld = gt_wld.swapaxis(2, 0)

        # get point cloud and surface normal
        save_ges_dir = os.path.join(
            save_dir, sub_names[sub_idx], ges_names[ges_names])
        os.mkdir(save_ges_dir)

        print(save_ges_dir)

        point_cloud_FPS = np.zeros(frame_num, sample_num, 6)
        volume_rotate = np.zeros(frame_num, 3, 3)
        volume_length = np.zeros(frame_num, 1)
        volume_offset = np.zeros(frame_num, 3)
        volume_GT_XYZ = np.zeros(frame_num, joint_num, 3)

        valid = msra_valid[sub_idx, ges_idx]

        for frm_idx in range(len(depth_files)):
            if not valid(frm_idx):
                continue
                
                # read binary files
            with open(ges_dir+str('%06d' %frm_idx)+'_depth.bin')
