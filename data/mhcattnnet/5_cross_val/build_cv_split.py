import os 
import argparse

import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold

def replace_str_with_num(s):
    if(s == 'Positive'):
        return 1
    else:
        return 0

def load_relevant_dataframe(path):
    df = pd.read_csv(path)
    d = df[['peptide', 'mhc_amino_acid', 'bind_class', 'mhc_allele']]
    d.bind_class = d.bind_class.apply(lambda x: replace_str_with_num(x))
    d = d.rename(columns={'bind_class': 'bind'})
    return d

def perform_kfold_cv(d):
    X = d[['peptide', 'mhc_amino_acid', 'mhc_allele']]
    y = d[['bind']]
    cv = KFold(n_splits=5, random_state=3, shuffle=True)
    ctr = 1
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        
        d_train = pd.concat([X_train, y_train], axis=1)
        d_train = d_train[['peptide', 'mhc_amino_acid', 'bind', 'mhc_allele']]
        d_train.to_csv(str(ctr) + '_train.csv', index=False)
        
        d_test = pd.concat([X_test, y_test], axis=1)
        d_test = d_test[['peptide', 'mhc_amino_acid', 'bind', 'mhc_allele']]
        d_test.to_csv(str(ctr) + '_test.csv', index=False)

        ctr += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build cross validation splits')
    parser.add_argument('-p', '--path', required=True, help='path to the class corpus')
    args = parser.parse_args()

    d = load_relevant_dataframe(args.path)
    perform_kfold_cv(d)
