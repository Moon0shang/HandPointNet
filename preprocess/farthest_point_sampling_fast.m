function sampled_idx = farthest_point_sampling_fast(point_cloud, sample_num)
% farthest point sampling
% point_cloud: Nx3


pc_num = size(point_cloud,1);

if pc_num <= sample_num
    sampled_idx = [1:pc_num]';
        % randi([min,max],m,n)生成在min~max范围内的随机m*n矩阵
    sampled_idx = [sampled_idx; randi([1,pc_num],sample_num-pc_num,1)];
else
    sampled_idx = zeros(sample_num,1);
    sampled_idx(1) = randi([1,pc_num]);
    
    cur_sample = repmat(point_cloud(sampled_idx(1),:),pc_num,1);
    diff = point_cloud - cur_sample;
        %sum(A,n)n=1 行求和，n=2列求和
    min_dist = sum(diff.*diff, 2);

    for cur_sample_idx = 2:sample_num
        %% find the farthest point
        [~, sampled_idx(cur_sample_idx)] = max(min_dist);
        
        if cur_sample_idx < sample_num
            % update min_dist
            valid_idx = (min_dist>1e-8);    % 根据条件返回逻辑值
            diff = point_cloud(valid_idx,:) - repmat(point_cloud(sampled_idx(cur_sample_idx),:),sum(valid_idx),1);
            min_dist(valid_idx,:) = min(min_dist(valid_idx,:), sum(diff.*diff, 2));
        end
    end
end
    %A=unique(a) 返回a中不相同元素的集合
sampled_idx = unique(sampled_idx);