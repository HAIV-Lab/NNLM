import argparse
import os
from trainer import Trainer
import numpy as np
from visual import Visual

parser = argparse.ArgumentParser()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--milestones', type=int, nargs='+', default=[116, 233], help='Milestones')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma')
    # parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')

    parser.add_argument('--voc_len',type=int, default=42020, help='voc number')
    parser.add_argument('--embedding_dim',type=int, default=1024, help='embedding size')
    parser.add_argument('--output_dim', type=int, default=64, help="output dim")
    parser.add_argument('--dstore_mmap',type=str, default='/data/zqh/NLP/adaptive-knn-mt/store/datastore/it_finetune')
    parser.add_argument('--dstore_size',type=int, default=3608731, help='datastore size')
    parser.add_argument('--use_cluster', type=bool, default=False, help="if use word cluster")
    parser.add_argument('--cluster_type', type=str, default='spectrum', help='cluster type')
    parser.add_argument('--class_num', type=int, default=42020, help='class number')
    
    # contrastive learning
    parser.add_argument('--K', type=int, default=1000, help='queue size')
    parser.add_argument('--m', type=float, default=0.999, help='momentum')

    # save
    parser.add_argument('--save_path', type=str, default=None, help='save checkpoint dir')
    # dataset
    args = parser.parse_args()
    return args

# 数据集处理，数据量比较大，标签类别过多
# 模型，模型是否可以使用Resnet，是否要在后面加一些层
# 损失和准确率基本不变，损失太小，需要大量负样本

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    args = get_args()
    # Visual(args)    
    
    trainer = Trainer(args)
    trainer.train()


