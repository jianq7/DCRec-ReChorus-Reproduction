import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.sequential.SASRec import SASRec
from utils import layers

class DCRec(SASRec):
    """
    DCRec：在SASRec基础上添加对比学习，cl_weight=0时退化为纯SASRec
    修复导入路径 + 对比学习逻辑
    """
    reader = 'SeqReader'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'cl_weight', 'aug_prob', 'temperature']

    @staticmethod
    def parse_model_args(parser):
        # 先解析SASRec的基础参数
        parser = SASRec.parse_model_args(parser)
        # 添加对比学习相关参数
        parser.add_argument('--aug_prob', type=float, default=0.2, help='Mask ratio for sequence augmentation')
        parser.add_argument('--cl_weight', type=float, default=0.05, help='Weight of contrastive loss')
        parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for InfoNCE loss')
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        # 对比学习参数
        self.aug_prob = args.aug_prob
        self.cl_weight = args.cl_weight
        self.tau = args.temperature

        # 仅在cl_weight>0时初始化projector
        self.projector = None

        if self.cl_weight > 0:
            self.projector = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size),
                nn.ELU(),
                nn.Linear(self.emb_size, self.emb_size)
            )
            # 初始化projector参数
            self.apply(self.init_weights)


    def init_weights(self, module):
        # 复用SASRec的初始化，额外初始化projector
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _augment_safe_mask(self, seqs, lengths):
        """序列增强：随机mask非最后一个有效物品"""
        aug_seqs = seqs.clone()
        batch_size, max_len = seqs.shape
        # 生成mask掩码：仅mask有效物品且非最后一个位置
        rand_matrix = torch.rand(aug_seqs.shape, device=self.device)
        mask_candidate = (rand_matrix < self.aug_prob) & (aug_seqs > 0)
        indices = torch.arange(max_len, device=self.device).expand(batch_size, max_len)
        last_indices = lengths.unsqueeze(1) - 1
        not_last = (indices != last_indices)
        final_mask = mask_candidate & not_last
        # 执行mask（置0）
        aug_seqs.masked_fill_(final_mask, 0)
        return aug_seqs
        




    def _info_nce_loss(self, z1, z2):
        """恢复原去偏InfoNCE损失计算逻辑"""
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # 计算相似度矩阵
        logits = torch.matmul(z1, z2.T) / self.tau
        # 正例相似度（对角线）
        pos_sim = torch.diag(logits)
        # 负例相似度和（排除自身）
        exp_logits = torch.exp(logits)
        mask = torch.eye(batch_size, device=self.device).bool()
        exp_logits = exp_logits.masked_fill(mask, 0)
        neg_sum = torch.sum(exp_logits, dim=1)
        # 去偏修正
        correction_factor = batch_size / (batch_size - 1) if batch_size > 1 else 1.0
        # 计算InfoNCE
        numerator = torch.exp(pos_sim)
        denominator = numerator + correction_factor * neg_sum + 1e-8
        loss = -torch.log(numerator / denominator).mean()
        return loss

    def get_sasrec_seq_emb(self, seq, len_seq):
        """
        完全复刻SASRecBase.forward中的序列嵌入生成逻辑（抽成独立方法，避免重复）
        :param seq: 序列 [batch_size, history_max]
        :param len_seq: 序列有效长度 [batch_size]
        :return: 序列的最终嵌入 [batch_size, emb_size]
        """
        batch_size, seq_len = seq.shape
        # 1. 物品嵌入 + 位置嵌入（和SASRec完全一致）
        valid_his = (seq > 0).long()
        his_vectors = self.i_embeddings(seq)

        # 位置编码：从最近的交互开始计数（SASRec的核心逻辑）
        position = (len_seq[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # 2. Self-attention（复用SASRec的transformer_block）
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 3. 取最后一个有效位置的嵌入（和SASRec一致）
        his_vector = his_vectors[torch.arange(batch_size).to(self.device), len_seq.long() - 1, :]
        return his_vector

    def calculate_loss(self, out_dict, batch=None):
        """
        最终版损失计算：完全适配SASRec源码的序列嵌入逻辑
        """

        # 1. 计算SASRec的基础推荐损失（父类需要out_dict['prediction']）
        rec_loss = super().loss(out_dict)

        # 2. 仅在训练且cl_weight>0时计算对比损失
        cl_loss = 0.0
        if self.training and self.cl_weight > 0 and self.projector is not None:
            seqs = batch['history_items']  # [batch_size, history_max]
            lengths = batch['lengths']  # [batch_size]
            batch_size, seq_len = seqs.shape

            # 生成两个增强序列
            aug_seq1 = self._augment_safe_mask(seqs, lengths)
            aug_seq2 = self._augment_safe_mask(seqs, lengths)


            mask_num1 = (seqs != aug_seq1).sum().item()
            mask_num2 = (seqs != aug_seq2).sum().item()


            # ========== 复用抽出来的类方法 ==========
            seq_emb1 = self.get_sasrec_seq_emb(aug_seq1, lengths)
            seq_emb2 = self.get_sasrec_seq_emb(aug_seq2, lengths)

            # 投影到对比学习空间 + 计算InfoNCE损失
            z1 = self.projector(seq_emb1)
            z2 = self.projector(seq_emb2)

            cl_loss = self._info_nce_loss(z1, z2)


        # 3. 总损失 = 推荐损失 + 对比损失 * 权重
        total_loss = rec_loss + self.cl_weight * cl_loss
        return total_loss

    def calculate_contrastive_loss(self, batch):
        """
        独立的对比损失计算方法（供Runner调用）
        """
        if not self.training or self.cl_weight <= 0 or self.projector is None:
            return 0.0

        seqs = batch['history_items']  # [batch_size, history_max]
        lengths = batch['lengths']  # [batch_size]
        batch_size, seq_len = seqs.shape

        # 生成两个增强序列
        aug_seq1 = self._augment_safe_mask(seqs, lengths)
        aug_seq2 = self._augment_safe_mask(seqs, lengths)


        mask_num1 = (seqs != aug_seq1).sum().item()
        mask_num2 = (seqs != aug_seq2).sum().item()


        # ========== 复用抽出来的类方法 ==========
        seq_emb1 = self.get_sasrec_seq_emb(aug_seq1, lengths)
        seq_emb2 = self.get_sasrec_seq_emb(aug_seq2, lengths)

        # 投影到对比学习空间 + 计算InfoNCE损失
        z1 = self.projector(seq_emb1)
        z2 = self.projector(seq_emb2)
        cl_loss = self._info_nce_loss(z1, z2)


        return cl_loss

    def loss(self, out_dict, batch=None):  # 加默认值 None → 兼容单参数调用
        if batch is None:
            return super().loss(out_dict)  # 单参数时调用父类
        else:
            return self.calculate_loss(out_dict, batch)


