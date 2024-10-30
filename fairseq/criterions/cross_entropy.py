# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import numpy as np
import torch
import logging

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.tgt_dict = task.target_dictionary
        self.class_numbers = len(self.tgt_dict)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        # TODO: 生成features
        # features = self.task.forward_and_get_hidden_state_step(sample, model)
        # net_output = model.forward_compact_network(features)
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def focal_loss(self, input_values, gamma=1.0):
        p = torch.exp(-input_values)
        loss = (1.0 - p) ** gamma * input_values
        return loss.mean()
    
    # def compute_contrastive_loss(self, model, net_output, sample, reduce=True):
    #     # TODO: 从knn 搜索的结果中找和target对应的key，作为正样本，其它作为负样本
    #     # 或者直接在datastore中找对应target的key（最好是中心的key）
    #     # 负样本是其它target的key（其它是哪些，是所有吗，还是挑选一个）
    #     # 有些样本没有正样本，目前的做法时跳过，
    #     # 内存不断增长
    #     tgt_index = net_output[-2]
    #     knn_keys = net_output[-3]
    #     compact_hidden = net_output[-4]
    #     target = model.get_targets(sample, net_output).view(-1)
    #     tgt_index = tgt_index.view(-1, tgt_index.size(-2))
    #     knn_keys = knn_keys.view(-1, model.decoder.knn_datastore.k, 1024)
    #     compact_hidden = compact_hidden.view(-1, 1024)
    #     # print("knn_keys shape:", knn_keys.shape)
    #     # print("compact_hidden shape:", compact_hidden.shape)

    #     data_len = len(knn_keys)
    #     # positive sample 
    #     positive_sample = None
    #     negitive_sample = None
    #     compute_hidden = None
    #     # 只保留既有正样本，也有负样本的情况 原样本 compute_hidden 
    #     for i in range(data_len):
    #         # 直接从knn 搜索的结果中查询
    #         temp_positive = knn_keys[i][tgt_index[i]==target[i]]
    #         temp_negitive = knn_keys[i][tgt_index[i]!=target[i]]
    #         if positive_sample is None:
    #             if len(temp_negitive) > 0 and len(temp_positive) > 0:
    #                 positive_sample = temp_positive.mean(dim=0).reshape(1, -1)
    #                 negitive_sample = temp_negitive.mean(dim=0).reshape(1, -1)
    #                 compute_hidden = compact_hidden[i].reshape(1, -1) 
    #         else:
    #             if len(temp_negitive) > 0 and len(temp_positive) > 0:
    #                 positive_sample = torch.cat((positive_sample, temp_positive.mean(dim=0).reshape(1, -1)))
    #                 negitive_sample = torch.cat((negitive_sample, temp_negitive.mean(dim=0).reshape(1, -1)))
    #                 compute_hidden = torch.cat((compute_hidden, compact_hidden[i].reshape(1, -1)))
        
    #     if compute_hidden is None:
    #         return 0.0

    #     batch_size = compute_hidden.shape[0]
    #     pos_dis = torch.exp((positive_sample * compute_hidden).sum(dim=1))
    #     neg_dis = torch.exp((negitive_sample * compute_hidden).sum(dim=1))
    #     con_loss = (-torch.log(pos_dis/(pos_dis + neg_dis))).sum()
    #     con_loss = con_loss / batch_size

    #     # print("con_loss:", con_loss)

    #     return con_loss

    def compute_contrastive_loss(self, model, net_output, sample, reduce=True):
        # TODO: 计算搜索得到的keys的中心，与查询样本计算距离，构造损失
        knn_keys = net_output[-3]
        compact_hidden = net_output[-4]
        compact_hidden = compact_hidden.reshape(-1, compact_hidden.shape[-1])
        knn_protos = torch.mean(knn_keys, dim=2).reshape(-1, knn_keys.shape[-1])
        
        mse_loss = F.mse_loss(compact_hidden, knn_protos, reduction="mean")
        return mse_loss


    def compute_loss(self, model, net_output, sample, reduce=True):
        # TODO: 利用knn的分布与compact_network的分布 计算损失
        con_loss = self.compute_contrastive_loss(model, net_output, sample, reduce=reduce)
        lprobs, compact_probs, nmt_probs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        
        knn_probs = net_output[2]
        tgt_index = net_output[-2]
        # print("tgt_index shape:", tgt_index.shape)

        compact_probs = compact_probs.view(-1, compact_probs.size(-1))
        tgt_index = tgt_index.view(-1, tgt_index.size(-2))
        knn_probs = knn_probs.view(-1, knn_probs.size(-1))
        nmt_probs = nmt_probs.view(-1, nmt_probs.size(-1))

        # print(tgt_index)

        index_knn_probs = None
        index_compact_probs = None
        data_len = len(knn_probs)
        for i in range(data_len):
            # print("tgt index:", tgt_index[i])
            if index_knn_probs is None:
                index_knn_probs = knn_probs[i][tgt_index[i]].reshape(1, -1)
                index_compact_probs = compact_probs[i][tgt_index[i]].reshape(1, -1)
            else:
                index_knn_probs = torch.cat((index_knn_probs, knn_probs[i][tgt_index[i]].reshape(1, -1))) 
                index_compact_probs = torch.cat((index_compact_probs, compact_probs[i][tgt_index[i]].reshape(1, -1)))

        
        # target_frequency = model.decoder.get_knn_frequency(target)
        # sel_index = np.random.randint(data_len) 
        # temp_loss = F.kl_div(index_compact_probs, index_knn_probs)
        # predicts = lprobs[sel_index].argmax(dim=-1)
        # knn_predicts = knn_probs[sel_index].argmax(dim=-1)
        # print("index_knn_probs: ", index_knn_probs[sel_index])
        # print("index_compact_probs: ", index_compact_probs[sel_index])
        # print("index_kl_loss:", temp_loss)
        # print("target word: ", self.tgt_dict[target[sel_index]]) 
        # print("target_frequency: ", target_frequency[sel_index])
        # print('predict word: ', self.tgt_dict[predicts])
        # print('knn predict word: ', self.tgt_dict[knn_predicts])
        # print()

        # compect_probs = torch.mm(torch.eye(len(target)*target_frequency, compect_probs))
        # knn_probs = torch.mm(torch.eye(len(target)).cuda().float() * target_frequency, knn_probs)

        # index_sel = np.random.randint(len(knn_probs))
        # print("knn probs: ", knn_probs[index_sel][knn_probs[index_sel]>0])
        # print("compect probs: ", compect_probs[index_sel][knn_probs[index_sel]>0])

        if len(knn_probs) !=0:
            comp_loss = F.kl_div(index_compact_probs, index_knn_probs, reduction="mean")

        
        
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="mean" if reduce else "none",
        )


        # logging.info(f"loss:{loss} comp_loss:{comp_loss} con_loss:{con_loss}")
        # print("comp_loss: ", comp_loss)
        # print("con_loss: ", con_loss)
        if len(knn_probs) !=0:
            total_loss = loss + comp_loss + con_loss
        else:
            total_loss = loss + con_loss

        return total_loss, total_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
