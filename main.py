import torch
from torch.utils.data import DataLoader
from Datasets import InitDataset, EpinionDataset

if __name__ == '__main__':
    ds = InitDataset()
    train_df, validation_df, test_df = ds.split_train_test(0.1, 0.1) # valid and test portion in 0-1
    
    batch_size = 1000 #
    train_ds = EpinionDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = EpinionDataset(validation_df)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_ds = EpinionDataset(test_df)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    print(next(iter(train_dl)))