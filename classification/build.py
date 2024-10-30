from torch.utils.data import DataLoader
from dataset import PrefixDataset, PrefixDataset1

def build_dataset(args):
    dataset = PrefixDataset1(args)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle = True
    )
    return dataloader