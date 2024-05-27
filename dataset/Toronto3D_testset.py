from tkinter import N
from utils.data_process import DataProcessing as DP
from utils.config import ConfigToronto3D as cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import random
import os
from helper_ply import read_ply

class Toronto3D:
    def __init__(self, mode='train', data_list=None, ignore_labels=None):
        self.data_list = None
        self.name = 'Toronto3D'
        self.mode = mode
        self.path = '/media/wcj/9C922B9C922B7A42/zl/Toronto_3D'
        self.label_to_names = {0: 'unclassified',
                               1: 'Ground',
                               2: 'Road marking',
                               3: 'Natural',
                               4: 'Building',
                               5: 'Utility line',
                               6: 'Pole',
                               7: 'Car',
                               8: 'Fence'}
        self.num_classes = len(self.label_to_names) - 1
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.full_pc_folder = join(self.path, 'original_ply')

        # Initial training-validation-testing files
        self.train_files = ['L001', 'L003', 'L004']
        self.val_files = ['L002']
        self.test_files = ['L002']

        self.val_split = 3

        self.train_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.train_files]
        self.val_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.val_files]
        self.test_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.test_files]

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = []
        self.min_possibility = []
        self.class_weight = []

        self.input_trees = []
        self.input_colors = []
        self.input_labels = []

        self.load_sub_sampled_clouds(cfg.sub_grid_size, mode)

    def load_sub_sampled_clouds(self, sub_grid_size, mode):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        if mode == 'test':
            files = self.test_files
            self.data_list = files
        else: 
            files = np.hstack((self.train_files, self.val_files))
            self.data_list = files

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            # read RGB / intensity accoring to configuration
            if cfg.use_rgb and cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'], data['intensity'])).T
            elif cfg.use_rgb and not cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            elif not cfg.use_rgb and cfg.use_intensity:
                sub_colors = data['intensity'].reshape(-1,1)
            else:
                sub_colors = np.ones((data.shape[0],1))
            if self.mode == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            if self.mode in ['train', 'val']:
                self.input_labels += [sub_labels]

            # Get test re_projection indices
            if self.mode == 'test':
                print('\nPreparing reprojection indices for {}'.format(cloud_name))
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        # Random initialize
        for i, tree in enumerate(self.input_trees):
            self.possibility += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

        if self.mode != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels), return_counts=True)
            self.class_weight += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        print('finished')
        return

    def get_class_weight(self):
        return DP.get_class_weights(self.path, self.data_list, self.num_classes)

    def __len__(self):
        if self.mode == "train":
            return 10000*2
        else:
            return cfg.val_steps * cfg.val_batch_size

    def __getitem__(self, item):
        selected_pc, selected_color, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen()
        return selected_pc, selected_color, selected_labels, selected_idx, cloud_ind

    def spatially_regular_gen(self):

        # Generator loop

        # Choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility))

        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[cloud_idx])

        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[cloud_idx].data, copy=False)

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        query_idx = self.input_trees[cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        query_idx = DP.shuffle_idx(query_idx)

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]
        queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
        queried_pc_colors = self.input_colors[cloud_idx][query_idx]
        if self.mode == 'test':
            queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
            queried_pt_weight = 1
        else:
            queried_pc_labels = self.input_labels[cloud_idx][query_idx]
            # queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
            queried_pt_weight = np.array([self.class_weight[0][n] for n in queried_pc_labels])

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
        self.possibility[cloud_idx][query_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))
        return queried_pc_xyz, queried_pc_colors, queried_pc_labels, query_idx, np.array([cloud_idx], dtype=np.int32)

    def tf_map(self, batch_pc, batch_color, batch_label, batch_pc_idx, batch_cloud_idx):
        features = np.dstack((batch_pc, batch_color))
        # features = [[[x y z r g b]...]]]
        input_points = []
        input_colors = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        # 用于存储不同层级下的输入数据
        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            # 根据配置中的子采样比率从点云数据中选取一部分子采样点，结果存储在sub_points中。

            sub_colors = batch_color[:, :batch_color.shape[1] // cfg.sub_sampling_ratio[i], :]

            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            # 将子采样点对应的最近邻点索引截取为与子采样点相同的形状，
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            # 在子采样点集中搜索与原始点云数据的最近邻点的索引

            input_points.append(batch_pc)
            input_colors.append(batch_color)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            batch_pc = sub_points
            batch_color = sub_colors
            # 更新batch_pc为子采样点集，用于下一层级的迭代。

        input_list = input_points + input_colors + input_neighbors + input_pools + input_up_samples
        # 5层input_points+5层....
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_color, selected_labels, selected_idx, cloud_ind = [], [], [], [], []
        # 遍历批次中的每个样本，并将对应的数据添加到相应的列表中。
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_color.append(batch[i][1])
            selected_labels.append(batch[i][2])
            selected_idx.append(batch[i][3])
            cloud_ind.append(batch[i][4])

        # 将列表中的数据转换为NumPy数组
        selected_pc = np.stack(selected_pc)
        selected_color = np.stack(selected_color)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_color, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())

        inputs['color'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['color'].append(torch.from_numpy(tmp).float())

        inputs['neigh_idx'] = []
        for tmp in flat_inputs[2 * num_layers: 3 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())

        inputs['sub_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())

        inputs['interp_idx'] = []
        for tmp in flat_inputs[4 * num_layers:5 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())

        inputs['features'] = torch.from_numpy(flat_inputs[5 * num_layers]).transpose(1, 2).float()
        # features = [xyzrgb]一次取出
        # 不经过transpose的shape=(1,65536,6)
        # 经过之后变为(1，6，65536)
        # 相当于[[x,y,z,r,g,b],[x,y,z,r,g,b],..]变为[[x,x,...],[y,y,..]]
        inputs['labels'] = torch.from_numpy(flat_inputs[5 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[5 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[5 * num_layers + 3]).long()

        return inputs
