# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadedAttention,LayerNorm
from torch.nn.init import xavier_uniform_
import copy

class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)    # 第一个嵌入层，学习item的嵌入向量，item_size=max_item + 1
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)          # 第二个嵌入层，学习位置的嵌入向量
        self.batch_size = args.batch_size

    # 对输入序列添加位置编码
    # 输入：sequence，形状是[batch, seq_len]
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)       # 获取序列的长度
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)   # 生成一个与序列长度相同的位置标识张量，表示每个位置的位置标识，形状是[seq_len]
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)                        # position_ids的形状变化：[seq_len] => [1, seq_len] => [batch, seq_len]
        item_embeddings = self.item_embeddings(sequence)                             # 获取item的嵌入向量
        position_embeddings = self.position_embeddings(position_ids)                 # 获取位置的嵌入向量
        sequence_emb = item_embeddings + position_embeddings                         # 向量相加
        sequence_emb = self.LayerNorm(sequence_emb)              # 虽然在父类的初始化方法中没有定义LayerNorm，但是在子类WTRecModel中定义了LayerNorm
        sequence_emb = self.dropout(sequence_emb)   # 应用LayerNorm和dropout层对嵌入进行进一步的规范化和防止过拟合

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    # not used 双向掩码
    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # 生成左对角的attention mask，即只能看到左边的item，和输入的形状不同（单向掩码）
    # 一般情况下，生成的掩码的形状和输入的形状是一样的（双向掩码）
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, input_ids, all_sequence_output=False):
        pass            # 由子类 BSARecModel 实现

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)       # 由子类 BSARecModel 实现的forward方法

    def calculate_loss(self, input_ids, answers):
        pass            # 由子类 BSARecModel 实现



class FilterMixerLayer(nn.Module):
    def __init__(self, hidden_size, i, args):
        super(FilterMixerLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.config = args    # SLIME4Rec.yaml
        self.max_seq_length = args.max_seq_length  # 50

        # complex_weight 存储复权重，形状 (1, freq_bins, hidden_size, 2)，最后的 2 代表实部和虚部
        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)        
        
        self.out_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.n_layers = args.num_hidden_layers   # 2
        self.residual = args.residual

        self.dynamic_ratio = args.dynamic_ratio  # 0.8 代表动态滤波占 80%，其余 20% 用于静态滤波
        # 滑动窗口的步长，标量
        self.slide_step = ((self.max_seq_length // 2 + 1) * (1 - self.dynamic_ratio)) // (self.n_layers - 1)

        self.static_ratio = 1 / self.n_layers   # 静态滤波
        self.filter_size = self.static_ratio * (self.max_seq_length // 2 + 1)
        G_i = i

        self.w = self.dynamic_ratio
        self.s = self.slide_step
            # left和right决定保留的频率范围
        self.left = int(((self.max_seq_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
        self.right = int((self.max_seq_length // 2 + 1) - G_i * self.s)

            

    def forward(self, input_tensor):
        # print("input_tensor", input_tensor.shape)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')   #输出是复数

        
        weight = torch.view_as_complex(self.complex_weight)
        x[:, :self.left, :] = 0
        x[:, self.right:, :] = 0
        output = x * weight

        sequence_emb_fft = torch.fft.irfft(output, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)

        if self.residual:
            origianl_out = self.LayerNorm(hidden_states + input_tensor)
        else:
            origianl_out = self.LayerNorm(hidden_states)

        return origianl_out

### 2024-11-9 新增一个短期兴趣编码器 👇
class ShortTermInterestsEncoder(nn.Module):
    def __init__(self, args):
        super(ShortTermInterestsEncoder, self).__init__()
        self.args = args
        self.rnn = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True)

        self.query_proj = nn.Linear(args.hidden_size, args.hidden_size)  # 用于生成查询向量的投影层
        self.key_proj = nn.Linear(args.hidden_size, args.hidden_size)  # key 投影层
        self.value_proj = nn.Linear(args.hidden_size, args.hidden_size)  # value 投影层

        self.attn_layer = MultiHeadedAttention(args)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor, attention_mask):
        # rnn_output的形状为[batch, seq_len, hidden_size]
        rnn_output, _ = self.rnn(input_tensor)


        # 2025.3.2 使用独立的查询向量
        query = self.query_proj(input_tensor)  # 查询向量
        key = rnn_output  # 键
        value = rnn_output  # 值
        attn_output = self.attn_layer(query, key, value, attention_mask)

        hidden_states = attn_output
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


###### 2025.2.14 SLIME4Rec模块
class FMBlock(nn.Module):
    def __init__(self, i, args):
        super(FMBlock, self).__init__()
        self.filter_mixer_layer = FilterMixerLayer(args.hidden_size, i, args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask=None):
        filter_mixer_layer_output = self.filter_mixer_layer(hidden_states)  # 不用掩码
        feedforward_output = self.feed_forward(filter_mixer_layer_output)

        return feedforward_output

### 长期兴趣编码器 FMRecEncoder
class FMRecEncoder(nn.Module):
    def __init__(self, args):
        super(FMRecEncoder, self).__init__()
        self.args = args
        # FMRecEncoder每一层可能会有不同的超参数配置，因为每个FMBlock(i, args)可能依赖于层的索引i
        self.blocks = nn.ModuleList([copy.deepcopy(FMBlock(i,args)) for i in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)  # 不用掩码
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            return all_encoder_layers[-1]  # 返回最后一层的输出

        return all_encoder_layers


class DuoRecModel(SequentialRecModel):
    def __init__(self, args):
        super(DuoRecModel, self).__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FMRecEncoder(args)      # 长期兴趣
        self.short_term_encoder = ShortTermInterestsEncoder(args)   # 短期兴趣
        # 学习权重
        self.weight_long_term = nn.Parameter(torch.tensor(0.5))
        self.weight_short_term = nn.Parameter(torch.tensor(0.5))

        self.batch_size = args.batch_size
        self.gamma = 1e-10

        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau     # 温度参数
        self.sim = args.sim    # 相似度计算方式，dot或cos
        self.lmd_sem = args.lmd_sem     #无监督损失权重
        # self.lmd = args.lmd     #有监督损失权重

        self.apply(self.init_weights)

    # 生成掩码矩阵，用于排除正样本对
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        # 对于 batch 中的每个样本 i，排除其对应的增强样本 i + batch_size
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask     # 返回一个布尔掩码矩阵，标记哪些位置应被视为负样本

    # 计算信息对比损失
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.

        计算InfoNCE损失所需的对数概率和标签(不显式采样负样本)

        参数:
            z_i (torch.Tensor): 第一个增强视图的特征张量，形状为 (batch_size, seq_len, feature_dim)
            z_j (torch.Tensor): 第二个增强视图的特征张量，形状为 (batch_size, seq_len, feature_dim)
            temp (float): 温度系数，用于缩放相似度得分
            batch_size (int): 原始批尺寸大小(每个增强视图的样本数)
            sim (str): 相似度计算方式，'cos'表示余弦相似度，'dot'表示点积相似度

        返回:
            logits (torch.Tensor): 包含正负样本相似度的矩阵，形状为 (2*batch_size, 1+2*(batch_size-1))
            labels (torch.Tensor): 全零标签向量，形状为 (2*batch_size,)
        """

        N = 2 * batch_size

        # 特征拼接与维度处理：将两个视图的特征拼接并取最后一个时间步的特征
        # 输出形状：z.shape = (2*batch_size, feature_dim)
        z = torch.cat((z_i, z_j), dim=0)
        z = z[:, -1, :]      # 仅保留序列最后一个位置的向量表示

        # 全量相似度矩阵计算：计算所有样本间的相似度(包含正样本对和负样本对)
        # 相似度矩阵形状：sim.shape = (2*batch_size, 2*batch_size)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        # 正样本相似度提取：从对角线偏移位置获取正样本对(i,j)和(j,i)的相似度
        # 结果形状：positive_samples.shape = (2*batch_size, 1)
        sim_i_j = torch.diag(sim, batch_size)    # 获取上对角线元素(i,j)
        sim_j_i = torch.diag(sim, -batch_size)   # 获取下对角线元素(j,i)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # 负样本掩码处理：生成仅包含负样本位置的掩码矩阵
        # 当batch_size变化时动态生成掩码，否则使用预存掩码
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        # 负样本相似度提取：通过掩码矩阵获取所有负样本相似度
        # 结果形状：negative_samples.shape = (2*batch_size, 2*(batch_size-1))
        negative_samples = sim[mask].reshape(N, -1)

        # 构造分类任务输出：正样本作为第0类，负样本作为后续类别
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        # 获取短期兴趣表示
        short_term_output = self.short_term_encoder(sequence_emb, extended_attention_mask)
        # 加权求和 LS-term
        sequence_output = self.weight_long_term * sequence_output + self.weight_short_term * short_term_output

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        """计算模型的总损失，包含交叉熵损失和多种对比学习损失

        Args:
            input_ids (Tensor): 输入序列的token id张量，形状为[batch_size, seq_len]
            answers (Tensor): 正样本答案id张量，形状为[batch_size]
            neg_answers (Tensor): 负样本答案id张量（当前代码中未使用，可能为后续扩展预留）
            same_target (Tensor): 语义增强后的等效序列张量，形状同input_ids
            user_ids (Tensor): 用户id张量（当前代码中未使用，可能为用户特征预留）

        Returns:
            Tensor: 包含交叉熵损失和对比学习损失的总损失标量值
        """

        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        # cross-entropy loss
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        
        # Unsupervised + Supervised NCE: dropout vs semantic augmentation
        # 混合增强对比分支（dropout增强 vs 语义增强）
        
        # unsupervised
        aug_seq_output = self.forward(input_ids)    # dropout增强表征
        # supervised
        sem_aug = same_target   # 语义增强表征
        sem_aug_seq_output = self.forward(sem_aug)

        sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)

        loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)     # 添加混合对比损失

        return loss
