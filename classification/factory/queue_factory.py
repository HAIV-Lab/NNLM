import torch
import torch.nn as nn
import torch.nn.functional as F

def create_queue(num_classes, dim, K, args):
    """
    num_classes: vocab size
    dim: embedding size after hidden
    K: queue size
    """
    queue = torch.randn(num_classes, dim, K)
    queue = F.normalize(queue, dim=1)
    queue_ptr = torch.zeros(num_classes, dtype=torch.long)
    dequeuer = MultiDequeuer(K, num_classes, args)
    return queue.cuda(), queue_ptr.cuda(), dequeuer

class Dequeuer(object):
    def __init__(self, K, num_classes, args) -> None:
        self.K = K
        self.num_classes = num_classes
        self.args = args
    
    def dequeue_and_enqueue(self, queue, queue_ptr, keys):
        raise NotImplementedError

class MultiDequeuer(Dequeuer):
    def __init__(self, K, num_classes, args) -> None:
        super().__init__(K, num_classes, args)
    
    def dequeue_and_enqueue(self, queue, queue_ptr, keys, cls_labels):
        for class_id in torch.unique(cls_labels):
            cls_keys = keys[cls_labels != class_id]
            batch_size = cls_keys.size(0)
            cls_idx = class_id
            ptr = int(queue_ptr[cls_idx])

            if ptr + batch_size >= self.K:
                queue[cls_idx][:, ptr:] = cls_keys.T[:, :self.K - ptr]
                queue[cls_idx][:, :(ptr + batch_size) % self.K] = cls_keys.T[:, self.K - ptr:]
            else:
                queue[cls_idx][:, ptr: ptr + batch_size] = cls_keys.T
            ptr = (ptr + batch_size) % self.K
            queue_ptr[cls_idx] = ptr
        return queue, queue_ptr