from torch.utils.data import Dataset
from random import sample
from tqdm import tqdm

import numpy as np
import torch
import math

labels = [2, 4, 5, 6, 9, 11]

class PrefixDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # 修改float16->float32
        self.data = np.memmap(self.args.dstore_mmap + '/keys.npy', dtype=np.float32,
                              mode='r', shape=(self.args.dstore_size, self.args.embedding_dim))
        self.label = np.memmap(self.args.dstore_mmap + '/vals.npy', dtype=np.int, 
                                mode='r',shape=(self.args.dstore_size, 1))
        self.data = np.asarray(self.data)
        self.label = np.asarray(self.label)
        self.label = self.label.reshape(self.label.shape[0])

        self.vocab_freq = [0 for _ in range(self.args.voc_len)]
        self.key_list = [[] for _ in range(self.args.voc_len)]

        # 统计频率并构造key_list
        for i in range(self.args.dstore_size):
            label = self.label[i]
            self.vocab_freq[label] += 1
            self.key_list[label].append(self.data[i])
        
        if self.args.use_cluster:
            ## inner clustering refine
            cluster_type = self.args.cluster_type
            min_samples = 4
            cluster_algorithm_list = ['spectrum', 'dbscan']
            assert cluster_type in cluster_algorithm_list, 'the cluster algorithm should be in the list: ' + ' '.join(cluster_algorithm_list)
            
            if cluster_type == 'spectrum':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_init=3, n_neighbors=min_samples)
            elif cluster_type == 'dbscan':
                from sklearn.cluster import DBSCAN
                sc = DBSCAN(eps=10, min_samples=min_samples)
            
            print('start clustering ...')
            new_key_list = []
            new_val_list = []
            base_number = min_samples
            # Limited by memory, 100000 koran/it/medical (<=10M) 20000 for law/subtitles (>=19M). 
            sample_bound = 100000
            for vocab_id, keys in enumerate(self.key_list):
                if len(keys) == 0:
                    continue
                
                print('clustering %d' % vocab_id)

                '''
                key_list[0] is a list of all-zero keys, because vocab[0] is '<s>'
                key_list[1~3] are not all-zero keys, of which the vocabs are '<pad> </s> <unk>'
                '''
                if vocab_id < 4 and vocab_id != 2:
                    continue

                if len(keys) <= base_number:
                    new_key_list.append(keys)
                    new_val_list.append([vocab_id for _ in range(len(keys))])
                    continue

                ## to decrease the computation
                if len(keys) > sample_bound:
                    keys = sample(keys, sample_bound)

                sc.n_clusters = int(math.log(len(keys)+base_number, base_number))
                sc.n_neighbors = min(len(keys), min_samples)
                print(f"sc.n_clusters:{sc.n_clusters} sc.n_neighbors:{sc.n_neighbors}")

                keys = np.array(keys)

                clustering = sc.fit(keys)
                labels = clustering.labels_

                tmp_key = [[] for _ in range(labels.max()+1)]
                for n in range(labels.shape[0]):
                    if labels[n] == -1:
                        continue
                    tmp_key[labels[n]].append(keys[n])
                    # print(labels[j], end=' ')
                tmp_key = [key for key in tmp_key if len(key) != 0]
                new_key_list.extend(tmp_key)

                tmp_val = [[vocab_id for _ in range(len(key))] for key in tmp_key]
                new_val_list.extend(tmp_val)
                assert len(tmp_key) == len(tmp_val)

            
            self.key_list = new_key_list
            self.val_list = new_val_list
            '''
            After target-side clustering, tokens of the same vocab may be split
            into different slices of this new_val_list, like:
            [
             [5,5,5], [5,5,5,5,5],
             [6,], [6,6,6,6], [6,6,6], [6,6],
             [7],
             [8,8,8,8], [8,8],
              ...
            ]
            '''

            print('we get %d clusters' % len(self.key_list))

            # # post-processing
            # for i in range(len(self.key_list)):
            #     if len(self.key_list[i]) == 0:
            #         continue
            #     self.key_list[i] = np.array(self.key_list[i])

            print('cluster done. Get %d nodes' % sum([len(keys) for keys in self.key_list]))

        self.larger_than_2_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 2 ]
        self.larger_than_1_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 1 ]
        assert len(self.larger_than_2_vocab) > 0, 'the datastore is too sparse to conduct a good baseline'

        ## add up the cluster centroid into the cluster
        for i, keys in enumerate(self.key_list):
            if len(keys) > 0:
                # 去掉了half
                self.key_list[i].append(torch.tensor(keys).float().mean(dim=0).numpy())
                # self.val_list[i].append(self.val_list[i][0])

    def __len__(self):
        return len(self.larger_than_2_vocab)

    def __getitem__(self, index):
        index = index % len(self.larger_than_2_vocab)
        index = self.larger_than_2_vocab[index]
        pivot_sample = self.key_list[index][-1]
        positive_sample = sample(self.key_list[index][:-1], 1)[0]

        # 只找一了一个负样本，是否可以考虑找多个
        while True:
            idx_neg = sample(self.larger_than_1_vocab, 1)[0]
            if idx_neg != index:
                break

        idx_neg_subidx = sample(range(len(self.key_list[idx_neg])), 1)[0]
        negative_sample = self.key_list[idx_neg][idx_neg_subidx]
        

        return {'negative_sample': negative_sample,
                'positive_sample': positive_sample,
                'pivot_sample': pivot_sample,
                'label':index}
    
class PrefixDataset1(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        # 修改float16->float32
        self.data = np.memmap(self.args.dstore_mmap + '/keys.npy', dtype=np.float16,
                              mode='r', shape=(self.args.dstore_size, self.args.embedding_dim))
        self.label = np.memmap(self.args.dstore_mmap + '/vals.npy', dtype=np.int, 
                                mode='r',shape=(self.args.dstore_size, 1))
        
        self.data = np.asarray(self.data)
        self.label = np.asarray(self.label)
        self.label = self.label.reshape(self.label.shape[0])

        self.sel_data = None
        self.sel_label = None
        for j, i in enumerate(labels):
            if self.sel_data is None:
                self.sel_data = self.data[self.label == i]
                self.sel_label = np.full(self.data[self.label == i].shape[0], j)
            else:
                self.sel_data = np.concatenate((self.sel_data, self.data[self.label == i]))
                self.sel_label = np.concatenate((self.sel_label, np.full(self.data[self.label == i].shape[0], j)))
        
        


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        embedding = self.data[index]
        embedding1 = self.data[index]
        label = self.label[index]

        return embedding, embedding1, label
    



