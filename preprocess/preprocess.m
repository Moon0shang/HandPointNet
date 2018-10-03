% create point cloud from depth image
% author: Liuhao Ge

clc;clear;close all;

dataset_dir='D:/Cache/Git/HandPointNet/data/cvpr15_MSRAHandGestureDB/';%'../data/cvpr15_MSRAHandGestureDB/'
save_dir='./';
subject_names={'P0','P1','P2','P3','P4','P5','P6','P7','P8'};
gesture_names={'1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y'};

JOINT_NUM = 21;
SAMPLE_NUM = 1024;
sample_num_level1 = 512;
sample_num_level2 = 128;

load('msra_valid.mat');

for sub_idx = 1:length(subject_names)
    mkdir([save_dir subject_names{sub_idx}]);
    
    for ges_idx = 1:length(gesture_names)
        gesture_dir = [dataset_dir subject_names{sub_idx} '/' gesture_names{ges_idx}];
        depth_files = dir([gesture_dir, '/*.bin']);
        
        % 1. read ground truth
        fileID = fopen([gesture_dir '/joint.txt']);
            % fscanf(fileID,format,size) 依次读取文件内容
            %size 为数则表示读取数目，为矩阵则读取该大小的矩阵
        frame_num = fscanf(fileID,'%d',1);
        A = fscanf(fileID,'%f', frame_num*21*3);
        gt_wld=reshape(A,[3,21,frame_num]);
        gt_wld(3,:,:) = -gt_wld(3,:,:);     % gt_wld(3,:,:)数据*-1 
           % premute(A,order) 按照order的顺序重新排列矩阵
           % 此处gt_wld变成一个[fram_num,21,3]矩阵
        gt_wld=permute(gt_wld, [3 2 1]);
        
        fclose(fileID);
        
        % 2. get point cloud and surface normal
        save_gesture_dir = [save_dir subject_names{sub_idx} '/' gesture_names{ges_idx}];
        mkdir(save_gesture_dir);
        
        display(save_gesture_dir);      %显示路径 
            %初始化接受数据变量
        Point_Cloud_FPS = zeros(frame_num,SAMPLE_NUM,6);
        Volume_rotate = zeros(frame_num,3,3);
        Volume_length = zeros(frame_num,1);
        Volume_offset = zeros(frame_num,3);
        Volume_GT_XYZ = zeros(frame_num,JOINT_NUM,3);
            % valid = msra_valid 对应行列的值，为num * 1 矩阵
        valid = msra_valid{sub_idx, ges_idx};
        
        %从valid 中找到对应的值（逻辑值）
        for frm_idx = 1:length(depth_files)
                % 当frm_idx值为0时跳过接下来的代码并开始下一次循环
            if ~valid(frm_idx)      
                continue;
            end
            %% 2.1 read binary file
            fileID = fopen([gesture_dir '/' num2str(frm_idx-1,'%06d'), '_depth.bin']);
            img_width = fread(fileID,1,'int32');
            img_height = fread(fileID,1,'int32');

            bb_left = fread(fileID,1,'int32');
            bb_top = fread(fileID,1,'int32');
            bb_right = fread(fileID,1,'int32');
            bb_bottom = fread(fileID,1,'int32');
            bb_width = bb_right - bb_left;
            bb_height = bb_bottom - bb_top;

            valid_pixel_num = bb_width*bb_height;

            hand_depth = fread(fileID,[bb_width, bb_height],'float32');
            hand_depth = hand_depth';   % 转置
            
            fclose(fileID);
            
            %% 2.2 convert depth to xyz
            fFocal_MSRA_ = 241.42;	% mm  焦距
            hand_3d = zeros(valid_pixel_num,3);
            for ii=1:bb_height
                for jj=1:bb_width
                    idx = (jj-1)*bb_height+ii;
                        % hand_3d(idx, 1)=。。 相当于指定第idx行，1列的数值
                    hand_3d(idx, 1) = -(img_width/2 - (jj+bb_left-1))*hand_depth(ii,jj)/fFocal_MSRA_;
                    hand_3d(idx, 2) = (img_height/2 - (ii+bb_top-1))*hand_depth(ii,jj)/fFocal_MSRA_;
                    hand_3d(idx, 3) = hand_depth(ii,jj);
                end
            end

            valid_idx = 1:valid_pixel_num;
                % 寻找不为0 的点
            valid_idx = valid_idx(hand_3d(:,1)~=0 | hand_3d(:,2)~=0 | hand_3d(:,3)~=0);
            hand_points = hand_3d(valid_idx,:);
                % 去除只有一列或一行的元素。
            jnt_xyz = squeeze(gt_wld(frm_idx,:,:));
            
            %% 2.3 create OBB
                %PCA降维后的每个样本的特征的维数，一般不会超过训练样本的个数
                %COEFF是X矩阵所对应的协方差阵V的所有特征向量组成的矩阵，即变换矩阵或称投影矩阵，
                %COEFF每列对应一个特征值的特征向量，列的排列顺序是按特征值的大小递减排序，
                %返回的SCORE是对主分的打分，也就是说原X矩阵在主成分空间的表示。
                %SCORE每行对应样本观测值，每列对应一个主成份(变量)，它的行和列的数目和X的行列数目相同。
                %返回的latent是一个向量，它是X所对应的协方差矩阵的特征值向量。
            [coeff,score,latent] = pca(hand_points);
                % hand_points为一个n行3列的矩阵，所以coeff为一个3行3列的矩阵
            if coeff(2,1)<0
                coeff(:,1) = -coeff(:,1);
            end
            if coeff(3,3)<0
                coeff(:,3) = -coeff(:,3);
            end
            coeff(:,2)=cross(coeff(:,3),coeff(:,1));    %cross 计算叉乘

            ptCloud = pointCloud(hand_points);

            hand_points_rotate = hand_points*coeff;

            %% 2.4 sampling
                %size(n,1),返回矩阵的行数或列数，1为行数，2为列数
            if size(hand_points,1)<SAMPLE_NUM
                    %floor(x) 函数将x中元素取整，值y为不大于本身的最小整数。对于复数，分别对实部和虚部取整
                tmp = floor(SAMPLE_NUM/size(hand_points,1));
                rand_ind = [];
                for tmp_i = 1:tmp
                    rand_ind = [rand_ind 1:size(hand_points,1)];
                end
                    % randperm(n,k) 随机排序，在n 范围内随机选取k个数随机排序
                    % mod(a,b) 取模运算，a/b取余，余数符号和被除数b相同
                rand_ind = [rand_ind randperm(size(hand_points,1), mod(SAMPLE_NUM, size(hand_points,1)))];
            else
                rand_ind = randperm(size(hand_points,1),SAMPLE_NUM);
            end
            hand_points_sampled = hand_points(rand_ind,:);
            hand_points_rotate_sampled = hand_points_rotate(rand_ind,:);
            
            %% 2.5 compute surface normal
            normal_k = 30;
                %pcnormals(ptcloud,k) 估计点云的法线
                % k 是局部平面拟合的点数，k >= 3
            normals = pcnormals(ptCloud, normal_k);
            normals_sampled = normals(rand_ind,:);

            sensorCenter = [0 0 0];
            for k = 1 : SAMPLE_NUM
               p1 = sensorCenter - hand_points_sampled(k,:);
               % Flip(翻转) the normal vector if it is not pointing towards the sensor.
               angle = atan2(norm(cross(p1,normals_sampled(k,:))),p1*normals_sampled(k,:)');
               if angle > pi/2 || angle < -pi/2
                   normals_sampled(k,:) = -normals_sampled(k,:);
               end
            end
            normals_sampled_rotate = normals_sampled*coeff;

            %% 2.6 Normalize Point Cloud
            x_min_max = [min(hand_points_rotate(:,1)), max(hand_points_rotate(:,1))];
            y_min_max = [min(hand_points_rotate(:,2)), max(hand_points_rotate(:,2))];
            z_min_max = [min(hand_points_rotate(:,3)), max(hand_points_rotate(:,3))];

            scale = 1.2;
            bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
            bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
            bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
            max_bb3d_len = bb3d_x_len;

            hand_points_normalized_sampled = hand_points_rotate_sampled/max_bb3d_len;
            if size(hand_points,1)<SAMPLE_NUM
                offset = mean(hand_points_rotate)/max_bb3d_len;
            else
                offset = mean(hand_points_normalized_sampled);
            end
                %repmat(A, r1,r2) 将A*(可以是矩阵)扩展成为r1 * r2 大小的矩阵
            hand_points_normalized_sampled = hand_points_normalized_sampled - repmat(offset,SAMPLE_NUM,1);

            %% 2.7 FPS Sampling
            pc = [hand_points_normalized_sampled normals_sampled_rotate];
            % 1st level
            sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1)';
            other_idx = setdiff(1:SAMPLE_NUM, sampled_idx_l1);
            new_idx = [sampled_idx_l1 other_idx];
            pc = pc(new_idx,:);
            % 2nd level
            sampled_idx_l2 = farthest_point_sampling_fast(pc(1:sample_num_level1,1:3), sample_num_level2)';
            other_idx = setdiff(1:sample_num_level1, sampled_idx_l2);
            new_idx = [sampled_idx_l2 other_idx];
            pc(1:sample_num_level1,:) = pc(new_idx,:);
            
            %% 2.8 ground truth
            jnt_xyz_normalized = (jnt_xyz*coeff)/max_bb3d_len;
            jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);

            Point_Cloud_FPS(frm_idx,:,:) = pc;
            Volume_rotate(frm_idx,:,:) = coeff;
            Volume_length(frm_idx) = max_bb3d_len;
            Volume_offset(frm_idx,:) = offset;
            Volume_GT_XYZ(frm_idx,:,:) = jnt_xyz_normalized;
        end
        % 3. save files
        save([save_gesture_dir '/Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        save([save_gesture_dir '/Volume_rotate.mat'],'Volume_rotate');
        save([save_gesture_dir '/Volume_length.mat'],'Volume_length');
        save([save_gesture_dir '/Volume_offset.mat'],'Volume_offset');
        save([save_gesture_dir '/Volume_GT_XYZ.mat'],'Volume_GT_XYZ');
        save([save_gesture_dir '/valid.mat'],'valid');
    end
end