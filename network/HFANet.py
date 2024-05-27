import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
            #print(d_out,d_in)

        d_out = d_in
        #print("fin:",d_out,d_in)
        #exit()
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 4:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-5]
                d_out = 2 * self.config.d_out[-5]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points):
        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['color'][i], end_points['neigh_idx'][i])
            #TODO: try use random sample to smaple labels..
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################
        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)

        features = self.dropout(features)
        features = self.fc3(features)

        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, color, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, color, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out + 3, d_out//2)

        # self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out + 3, d_out)
        # ########for new #############w
        self.mlp_color1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_color2 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)

        self.mlp_fea_info1 = pt_utils.Conv2d(d_out + d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_fea_info2 = pt_utils.Conv2d(d_out + d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)

        self.mlp_fea_off_xyz_color1 = pt_utils.Conv2d(d_out//2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_fea_off_xyz_color2 = pt_utils.Conv2d(d_out//2, d_out // 2, kernel_size=(1, 1), bn=True)

        self.mlp_fea_enc1 = pt_utils.Conv2d(d_out, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_fea_enc2 = pt_utils.Conv2d(d_out, d_out // 2, kernel_size=(1, 1), bn=True)

        self.mlp_xyz_color_off1 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_xyz_color_off2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)

        self.mlp_xyz_color_en1 = pt_utils.Conv2d(d_out, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp_xyz_color_en2 = pt_utils.Conv2d(d_out, d_out // 2, kernel_size=(1, 1), bn=True)

        self.final1 = pt_utils.Conv2d(d_out + d_out//2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.final2 = pt_utils.Conv2d(d_out * 2, d_out, kernel_size=(1, 1), bn=True)

        self.mlp7 = pt_utils.Conv2d(20, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp7_2 = pt_utils.Conv2d(20, d_out // 2, kernel_size=(1, 1), bn=True)

        # ########for new #############

    def forward(self, xyz, color, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        # feature:8*8*65536*1
        # xyz:8*65536*3
        # neigh_idx:8*65536*16
        # make modifications as suggested by https://github.com/QingyongHu/RandLA-Net/issues/19
        f_xyz, xyz_exp_dis = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        xyz_exp_dis = xyz_exp_dis.permute((0, 3, 1, 2))
        # print(xyz_exp_dis.shape)
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples 8*10*65536*16
        # f_xyz1 = self.mlp1(f_xyz)   # 8*8*65536*16

        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        # 8*8*65536*1 -> 8*8*65536 -> 8*65536*8 -> 8*65536*16*8
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        #'''

        # ################################ for new ########################
        f_color, color_exp_dis = self.relative_color_encoding(color, neigh_idx)
        color_exp_dis = color_exp_dis.permute((0, 3, 1, 2))
        # print(color_exp_dis.shape)  # 4,1,65536,16
        f_color = f_color.permute((0, 3, 1, 2))
        f_xyz_color = torch.cat([f_xyz, f_color], dim=1)
        #print(f_xyz_color.shape)
        f_xyz_color1 = self.mlp7(f_xyz_color)

        # 生成feature_info
        BAA_neigh_feat = f_neighbours
        BAA_tile_feat = feature.repeat(1, 1, 1, 16) # 8*8*65536*16
        BAA_fea_dist = BAA_tile_feat - BAA_neigh_feat
        # 生成feature_dis
        feature_exp_dist = torch.unsqueeze(torch.mean(torch.abs(BAA_fea_dist), dim=1), dim=1)
        feature_exp_dist = torch.exp(-feature_exp_dist)
        # print(feature_exp_dist.shape)   #4,1,65536,16
        # end
        BAA_feat_info = torch.cat([BAA_fea_dist, BAA_neigh_feat, BAA_tile_feat], dim=1)  # B, N, k, d_out 8*24(8+8+8)*65536*16
        BAA_feat_info = self.mlp_fea_info1(BAA_feat_info) # 8*8*65536*16
        BAA_remain_feat_info = BAA_feat_info
        # end

        # 通过xyz和color偏移生成feat_encoding
        BAA_neigh_feat_offsets_xyz_color = self.mlp_fea_off_xyz_color1(f_xyz_color1) # 8*8*65536*16
        BAA_feat_info = torch.cat([BAA_neigh_feat_offsets_xyz_color, BAA_feat_info], dim=1)  # 8*16(8+8)*65536*16
        BAA_feat_encoding = self.mlp_fea_enc1(BAA_feat_info)    # 8*8*65536*16
        # end

        # 通过feature偏移生成xyz_encoding和color_encoding
        BAA_neigh_xyz_color_offsets = self.mlp_xyz_color_off1(BAA_remain_feat_info)  # 8*3*65536*16 dim=1=>16->3
        #BAA_neigh_xyz_color_offsets = self.mlp_xyz_color_off1(BAA_feat_encoding)  # 8*3*65536*16 dim=1=>16->3

        BAA_xyz_color_info = torch.cat([BAA_neigh_xyz_color_offsets, f_xyz_color1], dim=1)
        BAA_xyz_color_encoding = self.mlp_xyz_color_en1(BAA_xyz_color_info)
        #end

        BAA_f_concat = torch.cat(([BAA_feat_encoding, BAA_xyz_color_encoding]), dim=1)
        # qu diao shaung bian pian yi
        #BAA_f_concat = torch.cat(([BAA_feat_info, f_xyz_color1]), dim=1)
        BAA_exp_concat = torch.cat(([xyz_exp_dis, feature_exp_dist*0.1, color_exp_dis*0.1]), dim=1)
        BAA_f_exp_concat = torch.cat((BAA_f_concat, BAA_exp_concat), dim=1)

        'check'
        #BAA_f_max = BAA_f_concat.max(dim=-1, keepdims=True)[0]  # 4,16,65536,1
        BAA_f_pc_agg = self.att_pooling_1(BAA_f_exp_concat) # 4,8,65536,1
        #BAA_encoding = torch.cat([BAA_f_max, BAA_f_pc_agg], dim=1) # 4,24,65536,1
        #BAA_encoding = self.final1(BAA_encoding)

        # BAA_f_pc_agg = self.att_pooling_1(BAA_f_exp_concat)
        f_pc_agg = BAA_f_pc_agg
        # ################################ for new ########################

        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples

        # ################################ for new ########################
        f_xyz_color2 = self.mlp7_2(f_xyz_color)

        # 生成feature_info
        BAA_neigh_feat = f_neighbours
        BAA_tile_feat = feature.repeat(1, 1, 1, 16)  # 8*8*65536*16
        BAA_fea_dist = BAA_neigh_feat - BAA_tile_feat
        # 生成feature_dis
        feature_exp_dist = torch.unsqueeze(torch.mean(torch.abs(BAA_fea_dist), dim=1), dim=1)
        feature_exp_dist = torch.exp(-feature_exp_dist)
        # end
        BAA_feat_info = torch.cat([BAA_fea_dist, BAA_neigh_feat, BAA_tile_feat], dim=1)  # B, N, k, d_out 8*16(8+8)*65536*16
        BAA_feat_info = self.mlp_fea_info2(BAA_feat_info)
        BAA_remain_feat_info = BAA_feat_info
        # end

        # 通过xyz和color偏移生成feat_encoding
        BAA_neigh_feat_offsets_xyz_color = self.mlp_fea_off_xyz_color2(f_xyz_color2)  # 8*8*65536*16
        BAA_feat_info = torch.cat([BAA_neigh_feat_offsets_xyz_color, BAA_feat_info], dim=1)  # 8*16(8+8)*65536*16
        BAA_feat_encoding = self.mlp_fea_enc2(BAA_feat_info)  # 8*8*65536*16
        # end

        # 通过feature偏移生成xyz_encoding和color_encoding
        BAA_neigh_xyz_color_offsets = self.mlp_xyz_color_off2(BAA_remain_feat_info)  # 8*3*65536*16 dim=1=>16->3
        #BAA_neigh_xyz_color_offsets = self.mlp_xyz_color_off2(BAA_feat_encoding)  # 8*3*65536*16 dim=1=>16->3

        BAA_xyz_color_info = torch.cat([BAA_neigh_xyz_color_offsets, f_xyz_color2], dim=1)
        BAA_xyz_color_encoding = self.mlp_xyz_color_en2(BAA_xyz_color_info)
        # end

        BAA_f_concat = torch.cat(([BAA_feat_encoding, BAA_xyz_color_encoding]), dim=1)
        # qu diao shuang bian pian yi
        #BAA_f_concat = torch.cat(([BAA_feat_info, f_xyz_color2]), dim=1)
        BAA_exp_concat = torch.cat(([xyz_exp_dis, feature_exp_dist, color_exp_dis]), dim=1)
        BAA_f_exp_concat = torch.cat((BAA_f_concat, BAA_exp_concat), dim=1)

        'check'
        #BAA_f_max = BAA_f_concat.max(dim=-1, keepdims=True)[0]  # 4,16,65536,1
        BAA_f_pc_agg = self.att_pooling_2(BAA_f_exp_concat)  # 4,8,65536,1
        #BAA_encoding = torch.cat([BAA_f_max, BAA_f_pc_agg], dim=1)  # 4,24,65536,1
        #BAA_encoding = self.final2(BAA_encoding)
        f_pc_agg = BAA_f_pc_agg

        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        xyz_exp_dis = torch.exp(-relative_dis)
        # batch*npoint*nsamples*10
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature, xyz_exp_dis

    def relative_color_encoding(self, color, neigh_idx):
        neighbor_color = self.gather_neighbour(color, neigh_idx)  # batch*npoint*nsamples*3

        color_tile = color.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_color = color_tile - neighbor_color  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_color, 2), dim=-1, keepdim=True))
        # batch*npoint*nsamples*10
        color_exp_dis = torch.exp(-relative_dis)
        # relative_feature = torch.cat([relative_dis, relative_color, color_tile, neighbor_color], dim=-1)
        relative_feature = torch.cat([relative_dis, relative_color, color_tile, neighbor_color], dim=-1)

        return relative_feature, color_exp_dis

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        #print(features)
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg
