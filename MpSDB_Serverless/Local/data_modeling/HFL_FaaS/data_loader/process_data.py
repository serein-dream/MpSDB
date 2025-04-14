import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sqlalchemy import create_engine
def process_wine_data(rpath, wpath, spath):
    red_df = pd.read_csv(rpath, sep=';')
    white_df = pd.read_csv(wpath, sep=';')
    red_df['color'] = 0
    white_df['color'] = 1
    wine_df = pd.concat([white_df, red_df], ignore_index=True)
    wine_df = shuffle(wine_df)
    wine_df.to_csv(spath, index=False)
def engine(db_name):
    engine = create_engine(f'ADD_by_yourself', encoding='utf-8')
    return engine
def read_file_to_sql(db_name, csv_path):
    df = pd.read_csv(csv_path, encoding="utf8", sep=',', dtype={'code': str})
    table_name = csv_path.split('/')[-1].split('.')[0]
    df.to_sql(table_name, con=engine(db_name), if_exists='append', index=False)
def split_iid_data(df, size):
    df = shuffle(df)
    df_list = np.array_split(df, size)
    for i in range(size):
        spath = f'data/table_{i + 1}.csv'
        df_list[i].to_csv(spath, index=False)
if __name__ == '__main__':
    red_wine_path = 'data/winequality-red.csv'
    white_wine_path = 'data/winequality-white.csv'
    save_path = 'data/wine_quality.csv'
    process_wine_data(red_wine_path, white_wine_path, save_path)
    read_file_to_sql('database_1', save_path)
    wine_df = pd.read_csv(save_path, encoding="utf8", sep=',', dtype={'code': str})
    split_iid_data(wine_df, 3)
    for i in range(3):
        read_file_to_sql(f'database_{i+1}', f'data/wine_quality_{i+1}.csv')