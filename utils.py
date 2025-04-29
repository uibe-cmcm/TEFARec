import time
import pandas as pd
import torch
from torch.utils.data import Dataset

def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

def predict_mse(model, dataloader, device):
    mse, sample_count, mae = 0, 0, 0
    user_ids = []
    item_ids = []
    ratings_list = []
    predicts = []
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings, X, Y = map(lambda x: x.to(device), batch)
            predict = model(user_id, item_id, X, Y)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            mae += torch.nn.functional.l1_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)

            user_ids.extend(user_id.cpu().numpy())
            item_ids.extend(item_id.cpu().numpy())
            ratings_list.extend(ratings.squeeze().cpu().numpy())
            predicts.extend(predict.squeeze().cpu().detach().numpy())

    return mse / sample_count, mae / sample_count

class TEFARecDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        if 'dm' in data_path:
            self.matrix = pd.read_csv(data_path)
            try:
                self.matrix = self.matrix.drop(columns=['Lyrics', 'Melody', 'Creativity'])
            except:
                print('no Lyrics col')
            self.matrix.iloc[:, -5:] = self.matrix.iloc[:, -5:].where(self.matrix.iloc[:, -5:] < 0, 1).where(self.matrix.iloc[:, -5:] > 0, -1)########
            self.X = torch.load('dm_Fu.pth')
            self.Y = torch.load('dm_Fi.pth')
        elif 'yelp' in data_path:
            self.matrix = pd.read_csv(data_path)
            self.X = torch.load('yelp_Fu.pt')
            self.Y = torch.load('yelp_Fi.pt')
        df = self.matrix[['userID', 'itemID', 'review', 'rating']]
        self.sparse_idx = set()
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]
        self.matrix = self.matrix.values

    def __getitem__(self, idx):
        user_id = self.matrix[idx, 0]
        item_id = self.matrix[idx, 1]
        return user_id, item_id, self.rating[idx], self.X[user_id], self.Y[item_id]

    def __len__(self):
        return self.rating.shape[0]
