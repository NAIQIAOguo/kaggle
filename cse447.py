import pandas as pd
import re
import numpy as np

def load(file_name):
    df_test = pd.read_csv(file_name)
    sLength = len(df_test['id'])
    df_test.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_test = df_test.fillna("miss")
    df_test['name'] = df_test['name'].str.lower()
    df_test['lvl1'] = df_test['lvl1'].str.lower()
    df_test['lvl2'] = df_test['lvl2'].str.lower()
    df_test['lvl3'] = df_test['lvl3'].str.lower()
    df_test['descrption'] = df_test['descrption'].str.lower()
    df_test['type'] = df_test['type'].str.lower()
    df_test['score'] = pd.Series(np.zeros(sLength, dtype=np.int), index=df_test.index)
    return df_test

def loadY(file_name):
    df = pd.read_csv(file_name)
    return df

def loadX(file_name):
    df_train = pd.read_csv(file_name)
    df_label = pd.read_csv('train_label.csv')
    df_train = df_train.set_index('id').join(df_label.set_index('id'), how='inner')
    df_train.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_train['name'] = df_train['name'].str.lower()
    df_train['lvl1'] = df_train['lvl1'].str.lower()
    df_train['lvl2'] = df_train['lvl2'].str.lower()
    df_train['lvl3'] = df_train['lvl3'].str.lower()
    df_train['descrption'] = df_train['descrption'].str.lower()
    df_train['type'] = df_train['type'].str.lower()

    # df['score'] = df_label['score']
    df_train = df_train.fillna("miss")

    # df.dropna(inplace=True)
    # df['score'] = pd.Series(df_label['score'], index=df.index)
    return df_train
