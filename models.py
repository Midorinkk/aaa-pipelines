import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import implicit
from scipy.sparse import coo_matrix

import numpy as np
import pandas as pd

import mlflow

import optuna

import os 
import random

from tqdm.auto import tqdm


DEFAULT_RANDOM_SEED = 123

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)


class RecDataset(Dataset):
    def __init__(self, users, items, item_per_users):
        self.users = users
        self.items = items
        self.item_per_users=item_per_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        user = self.users[i]
        return torch.tensor(user), torch.tensor(self.items[i]), self.item_per_users[user]
    

class LatentFactorModel(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes):
        super(LatentFactorModel, self).__init__()
        self.edim = edim
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)

    def forward(self, users, items):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items(items)
        res = torch.einsum('be,bne->bn', user_embedings, item_embedings)
        return res 

    def pred_top_k(self, users, K=10):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items.weight
        res = torch.einsum('ue,ie->ui', user_embedings, item_embedings)
        return torch.topk(res, K, dim=1)
    

def collate_fn(batch, num_negatives, num_items):
    users, target_items, users_negatives = [],[], []
    for triplets in batch:
        user, target_item, seen_item = triplets
        
        users.append(user)
        target_items.append(target_item)
        user_negatives = []
        
        while len(user_negatives)< num_negatives:
            candidate = random.randint(0, num_items)
            if candidate not in seen_item:
                user_negatives.append(candidate)
                
        users_negatives.append(user_negatives)
                
    
    positive = torch.ones(len(batch), 1)       
    negatives = torch.zeros(len(batch), num_negatives)
    labels = torch.hstack([positive, negatives])
    # print(torch.tensor(target_items))
    # print(users_negatives)
    items = torch.hstack([torch.tensor(target_items).reshape(-1, 1), torch.tensor(users_negatives)])
    return torch.hstack(users), items, labels


def calc_hitrate(df_preds, K):
    return  df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].max().mean()


def calc_prec(df_preds, K):
    return  (df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].mean()).mean()


def run_LFM(df: pd.DataFrame,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            user2seen: dict,
            user_indes: list,
            node_indes: list,
            model_name: str='baseline',
            run_name: str='baseline',
            BATCH_SIZE: int = 50_000,
            NUM_NEGATIVES: int = 5,
            EDIM: int = 128,
            EPOCH: int = 10,
            OPTIMIZER_NAME: str = 'Adam',
            LR: float = 1.0):
    
    seedEverything()

    train_dataset = RecDataset(df_train['user_index'].values, df_train['node_index'], user2seen)
    dataloader = DataLoader(train_dataset, shuffle=True,
                            num_workers=0, batch_size=BATCH_SIZE,
                            collate_fn=lambda x: collate_fn(x, NUM_NEGATIVES, max(df['node_index'].values)))


    model = LatentFactorModel(EDIM, user_indes, node_indes)

    if OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), LR)
    elif OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), LR)
    elif OPTIMIZER_NAME == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), LR)
    elif OPTIMIZER_NAME == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), LR)
    elif OPTIMIZER_NAME == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), LR)
    
    bar = tqdm(total = EPOCH )

    for i in range(EPOCH):
        bar_loader = tqdm(total = len(dataloader) ,)
        losses = []
        for i in dataloader:
            users, items, labels = i
            optimizer.zero_grad()
            logits = model(users, items)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            loss.backward()
            optimizer.step()
            bar_loader.update(1)
            bar_loader.set_description(f'batch loss - {loss.item()}')
            losses.append(loss.item())
        
        bar.update(1)
        bar.set_description(f'epoch loss - {sum(losses)/len(losses)}')


    K = 100
    test_users = df_test['user_index'].unique()

    preds = model.pred_top_k(torch.tensor(test_users), K)[1].numpy()
    df_preds = pd.DataFrame({'node_index': list(preds),
                            'user_index': test_users,
                            'rank': [[j for j in range(0, K)]for i in range(len(preds))]})

    df_preds = df_preds.explode(['node_index', 'rank']).merge(
        df_test[['user_index', 'node_index']].assign(relevant=1).drop_duplicates(),
        on = ['user_index', 'node_index'],
        how='left' ,
    )
    df_preds['relevant'] = df_preds['relevant'].fillna(0)

    prec = calc_prec(df_preds, 30)
    hitrate = calc_hitrate(df_preds, K)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {
                'hitrate': hitrate, 
                'prec_30': prec,
            }
        )
        mlflow.log_params(
            {
            'model_name': model_name,
            'edim': EDIM,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'epoch': EPOCH,
            'num_negatives': NUM_NEGATIVES,
            'optimizer_name': OPTIMIZER_NAME,
            }
        )

    return prec


def run_optuna_LFM(df: pd.DataFrame,
                   df_train: pd.DataFrame,
                   df_test: pd.DataFrame,
                   user2seen: dict,
                   user_indes: list,
                   node_indes: list,
                   model_name: str,
                   run_name: str,
                   OPTIMIZER_NAME: str):

    def objective(trial):
        seedEverything()

        # параметры
        EDIM = trial.suggest_categorical('EDIM', [32, 64, 128, 256])
        BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [2_000, 5_000, 10_000, 20_000, 25_000, 50_000])
        LR = trial.suggest_float('LR', 1e-3, 1, log=False)
        EPOCH = trial.suggest_int('EPOCH', 10, 30, step=5, log=False)
        NUM_NEGATIVES = trial.suggest_int('NUM_NEGATIVES', 5, 30, step=5, log=False)

        prec = run_LFM(df, df_train, df_test, user2seen,
                        user_indes, node_indes, model_name,
                        run_name, BATCH_SIZE, NUM_NEGATIVES,
                        EDIM, EPOCH, OPTIMIZER_NAME, LR)
        
        return prec
        
    sampler = optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED)

    study = optuna.create_study(study_name='LFM_tuning', direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=10)


def run_ALS_best(df: pd.DataFrame,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 model_name: str,
                 run_name: str
                 ):
    best_als_params = {'regularization': 0.01696469037727071,
                       'alpha': 11.047160266367834,
                       'iterations': 10,
                       'factors': 90}

    USERS = df['user_index'].unique().tolist()
    ITEMS = df['node_index'].unique().tolist()

    rows = df_train['user_index'].values
    cols = df_train['node_index'].values
    als_train_data = np.ones(df_train.shape[0])

    coo_train = coo_matrix((als_train_data, (rows, cols)), shape=(len(USERS), len(ITEMS)))
    csr_train = coo_train.tocsr()

    seedEverything()

    best_als_model = implicit.als.AlternatingLeastSquares(**best_als_params,
                                                         random_state=DEFAULT_RANDOM_SEED)
    best_als_model.fit(csr_train, show_progress=True)

    # тест
    K = 100
    test_users = df_test['user_index'].unique()


    ids, scores = best_als_model.recommend(test_users,
                                        csr_train[test_users],
                                        N=100,
                                        filter_already_liked_items=False)

    als_preds = pd.DataFrame({'node_index': list(ids),
                            'user_index': test_users,
                            'rank': [[j for j in range(0, K)]for i in range(len(ids))]})

    als_preds = als_preds.explode(['node_index', 'rank']).merge(
                df_test[['user_index', 'node_index']].assign(relevant=1).drop_duplicates(),
                on = ['user_index', 'node_index'],
                how='left' ,
    )
    als_preds['relevant'] = als_preds['relevant'].fillna(0)

    hitrate = calc_hitrate(als_preds, K)
    prec = calc_prec(als_preds, 30)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {
                'hitrate': hitrate, 
                'prec_30': prec,
            }
        )
        mlflow.log_params(
            {
            'model_name': model_name,
            'factors': best_als_params['factors'],
            'regularization': best_als_params['regularization'],
            'alpha': best_als_params['alpha'],
            'iterations': best_als_params['iterations']
            }
        )