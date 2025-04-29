import os
import time
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from config import Config
from model import TEFARec
from utils import TEFARecDataset, predict_mse, date

def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse, train_loss_explicit = predict_mse(model, train_dataloader, config.device)
    valid_mse, valid_ae_loss = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, ae loss {train_loss_explicit:.6f}, validation mse {valid_mse:.6f}, valid ae loss {valid_ae_loss:.6f}')
    start_time = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch, patience = 100, 0, 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_mae, total_loss_explicit, total_samples, count_neg_one = 0, 0, 0, 0, 0
        for batch in train_dataloader:
            user_id, item_id, ratings, X, Y = map(lambda x: x.to(config.device), batch)
            predict = model(user_id, item_id, X, Y)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            mae = F.l1_loss(predict.detach(), ratings.detach(), reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_mae += mae.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()
        valid_mse, valid_ae_loss = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        train_mae = total_mae / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; train mae {train_mae:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)
            patience = 0
        else:
            patience += 1
            if patience >=10:
                break

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss, mae = predict_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, test ae loss is {mae:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    def set_random_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_random_seed(42)
    config = Config()
    print(config)
    train_file = config.data_name + '_train.csv'
    valid_file = config.data_name + '_valid.csv'
    test_file = config.data_name + '_test.csv'

    train_dataset = TEFARecDataset(train_file, config)
    valid_dataset = TEFARecDataset(valid_file, config)
    test_dataset = TEFARecDataset(test_file, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    num_users = len(train_dataset.X)
    num_items = len(train_dataset.Y)
    model = TEFARec(config, num_users, num_items).to(config.device)

    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file))
