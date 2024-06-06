import pandas as pd
import numpy as np 

import json

import mlflow
import argparse

from models import *


def main(experiment_name,
         run_name,
         model_name,
         optimizer_name):
    seedEverything()

    mlflow.set_tracking_uri('http://84.201.128.89:90/')
    mlflow.set_experiment(experiment_name)

    with open('data/node2name.json', 'r') as f:
        node2name = json.load(f)

    node2name = {int(k):v for k,v in node2name.items()}

    df = pd.read_parquet('data/clickstream.parque')
    df = df.head(100_000)

    df['is_train'] = df['event_date']< df['event_date'].max() - pd.Timedelta('2 day')
    df['names'] = df['node_id'].map(node2name)

    train_cooks = df[df['is_train']]['cookie_id'].unique()
    train_items = df[df['is_train']]['node_id'].unique()


    df = df[(df['cookie_id'].isin(train_cooks)) & (df['node_id'].isin(train_items))]
    user_indes, index2user_id = pd.factorize(df['cookie_id'])
    df['user_index'] = user_indes

    node_indes, index2node = pd.factorize(df['node_id'])
    df['node_index'] = node_indes

    df_train, df_test = df[df['is_train']], df[~df['is_train']]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    user2seen = df_train.groupby('user_index')['node_index'].agg(lambda x: list(set(x)))

    if model_name == 'baseline':
        run_LFM(df, df_train, df_test, user2seen, user_indes, node_indes)
    elif model_name == 'baseline_tuning':
        run_optuna_LFM(df, df_train, df_test, user2seen, user_indes, node_indes,
                       model_name, run_name, optimizer_name)
    elif model_name == 'ALS_best':
        run_ALS_best(df, df_train, df_test, model_name, run_name)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', help='experiment_name', default = 'homework-pipeline-marakhimova')
    parser.add_argument('--run_name', help='run_name', default = 'baseline')

    parser.add_argument('--model_name', help='model_name', default = 'baseline')
    parser.add_argument('--optimizer_name', help='optimizer_name', default = 'Adam')

    args = parser.parse_args()
    main(experiment_name=args.experiment_name,
         model_name=args.model_name,
         run_name=args.run_name,
         optimizer_name=args.optimizer_name)
    
