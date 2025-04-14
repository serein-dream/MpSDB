import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from data_modeling.data_loader import engine
import pandas as pd
import numpy as np

class MysqlIterableDataSet(IterableDataset):
    def __init__(self, db_name, tb_name, cols):
        super(MysqlIterableDataSet, self).__init__()
        self.db_name = db_name
        self.tb_name = tb_name
        self.cols = cols
    def __iter__(self):
        conn = engine(self.db_name)
        data_iter = pd.read_sql(self.tb_name, con=conn, chunksize=1, columns=self.cols)
        for data in data_iter:
            tensor = torch.from_numpy(np.array(data, dtype=np.float32).flatten())
            x, y = tensor[:-1], tensor[-1]
            yield x, y

class MysqlDataSet(Dataset):
    def __init__(self, db_name, tb_name, cols, db_ip):
        self.db_name = db_name
        self.tb_name = tb_name
        self.cols = cols
        self.db_ip = db_ip
        conn = engine(self.db_name, db_ip)
        data = pd.read_sql(self.tb_name, con=conn, columns=self.cols)
        self.data = torch.from_numpy(data.values).float()
    def __getitem__(self, index):
        x = self.data[index][:-1]
        y = self.data[index][-1]
        return x, y
    def __len__(self):
        conn = engine(self.db_name, self.db_ip)
        length = pd.read_sql(f'SELECT COUNT(*) FROM {self.db_name}.{self.tb_name}', con=conn).iloc[0, 0]
        return length

if __name__ == "__main__":
    cols_list = ['index', 'fixed acidity', 'volatile acidity', 'citric acid',
                 'residual sugar', 'chlorides', 'free sulfur dioxide',
                 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                 'quality', 'color']
    mysql_dataset = MysqlDataSet("database_1", "wine_quality", cols_list, "localhost")
    train_size = int(0.7 * len(mysql_dataset))
    test_size = len(mysql_dataset) - train_size
    train_sets, test_sets = torch.utils.data.random_split(mysql_dataset, [train_size, test_size])
    dataloader = DataLoader(train_sets, batch_size=8)
    for batch in dataloader:
        x, y = batch