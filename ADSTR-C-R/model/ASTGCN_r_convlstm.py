# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ConvLSTM import ConvLstm
# from model.Transformer import greedy_decode
from lib.utils import scaled_Laplacian, cheb_polynomial

class AVWGCN(nn.Module):
    """
    自适应动态图卷积分支 (Adaptive Graph Convolution)
    """

    def __init__(self, in_channels, out_channels):
        super(AVWGCN, self).__init__()

        self.weights = nn.Parameter(
            torch.FloatTensor(in_channels, out_channels)
        )
        self.biases = nn.Parameter(
            torch.FloatTensor(out_channels)
        )

    def forward(self, x, adj):
        """
        x: (B, N, F_in, T)
        adj: (N, N)
        """

        batch_size, num_nodes, in_channels, num_timesteps = x.shape

        # (B, T, N, F)
        x_tmp = x.permute(0, 3, 1, 2)

        # 动态邻接矩阵聚合
        x_adp = torch.matmul(adj, x_tmp)

        # 特征变换
        x_adp = torch.matmul(x_adp, self.weights) + self.biases

        # 恢复维度
        return x_adp.permute(0, 2, 3, 1)


class Spatial_Attention_layer(nn.Module):#空间注意力层类
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):#前向传播
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):#带空间注意力的切比雪夫卷积类
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):#时间注意力层类
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class ASTGCN_block(nn.Module):

    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_of_vertices,
        num_of_timesteps,
        embed_dim=10
    ):
        super(ASTGCN_block, self).__init__()

        self.TAt = Temporal_Attention_layer(
            DEVICE,
            in_channels,
            num_of_vertices,
            num_of_timesteps
        )

        self.SAt = Spatial_Attention_layer(
            DEVICE,
            in_channels,
            num_of_vertices,
            num_of_timesteps
        )

        # 静态图支路
        self.cheb_conv_SAt = cheb_conv_withSAt(
            K,
            cheb_polynomials,
            in_channels,
            nb_chev_filter
        )

        # 动态图支路
        self.node_embeddings1 = nn.Parameter(
            torch.randn(num_of_vertices, embed_dim).to(DEVICE),
            requires_grad=True
        )

        self.node_embeddings2 = nn.Parameter(
            torch.randn(num_of_vertices, embed_dim).to(DEVICE),
            requires_grad=True
        )

        self.adaptive_conv = AVWGCN(
            in_channels,
            nb_chev_filter
        ).to(DEVICE)

        self.adaptive_weight = nn.Parameter(
            torch.tensor([0.5]).to(DEVICE),
            requires_grad=True
        )

        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1)
        )

        self.convlstm = ConvLstm(
            3,
            5,
            3,
            1,
            False,
            True,
            False
        )

        self.residual_conv = nn.Conv2d(
            in_channels,
            nb_time_filter,
            kernel_size=(1, 1),
            stride=(1, time_strides)
        )

        self.W1 = nn.Parameter(
            torch.FloatTensor(
                num_of_vertices,
                num_of_timesteps
            ).to(DEVICE)
        )

        self.W2 = nn.Parameter(
            torch.FloatTensor(
                num_of_vertices,
                num_of_timesteps
            ).to(DEVICE)
        )

        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # 1. Temporal Attention
        temporal_At = self.TAt(x)

        x_TAt = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps),
            temporal_At
        ).reshape(
            batch_size,
            num_of_vertices,
            num_of_features,
            num_of_timesteps
        )

        # 2. Spatial Attention
        spatial_At = self.SAt(x_TAt)

        # 3. 混合图卷积

        # 静态图卷积
        spatial_gcn_static = self.cheb_conv_SAt(
            x,
            spatial_At
        )

        # 动态图卷积
        adp_adj = F.softmax(
            F.relu(
                torch.mm(
                    self.node_embeddings1,
                    self.node_embeddings2.transpose(0, 1)
                )
            ),
            dim=1
        )

        spatial_gcn_dynamic = self.adaptive_conv(
            x,
            adp_adj
        )

        # 动态权重融合
        alpha = torch.sigmoid(
            self.adaptive_weight
        ).to(x.device)

        spatial_gcn = (
            alpha * spatial_gcn_static
            + (1 - alpha) * spatial_gcn_dynamic
        )

        # 4. 时间卷积
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3)
        )

        # 5. R-C-R结构

        x_residual = self.residual_conv(
            x.permute(0, 2, 1, 3)
        )

        time_conv_output = (
            time_conv_output
            + x_residual * self.W1
        ).unsqueeze(2)

        output = self.convlstm(
            time_conv_output.permute(
                3, 0, 2, 1, 4
            )
        )

        output = (
            output[0][0]
        )[:, :, -1, :, :].permute(
            0, 2, 1, 3
        )

        x_residual = self.ln(
            F.relu(
                x_residual * self.W2 + output
            ).permute(0, 3, 2, 1)
        ).permute(
            0, 2, 3, 1
        )

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(
        self,
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices
    ):
        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([
            ASTGCN_block(
                DEVICE,
                in_channels,
                K,
                nb_chev_filter,
                nb_time_filter,
                time_strides,
                cheb_polynomials,
                num_of_vertices,
                len_input
            )
        ])

        self.BlockList.extend([
            ASTGCN_block(
                DEVICE,
                nb_time_filter,
                K,
                nb_chev_filter,
                nb_time_filter,
                1,
                cheb_polynomials,
                num_of_vertices,
                len_input // time_strides
            )
            for _ in range(nb_block - 1)
        ])

        self.final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict,
            kernel_size=(1, nb_time_filter)
        )

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):

        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(
            x.permute(0, 3, 1, 2)
        )[:, :, :, -1].permute(
            0, 2, 1
        )

        return output

    def initialize(self):

        for m in self.modules():
            for p in m.parameters():

                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)


def make_model(
    DEVICE,
    nb_block,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    num_of_vertices,
    embed_dim=10
):

    L_tilde = scaled_Laplacian(adj_mx)

    cheb_polynomials = [
        torch.from_numpy(i)
        .type(torch.FloatTensor)
        .to(DEVICE)
        for i in cheb_polynomial(L_tilde, K)
    ]

    model = ASTGCN_submodule(
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices
    )

    for p in model.parameters():

        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

