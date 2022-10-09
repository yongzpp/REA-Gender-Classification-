from config import cfg
from optimize import Optimizer
from preprocess import Preprocessor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def read(path):
    '''
    Read the csv file in config file and clean features
    :params path: file path to training data
    :return cleaned dataframe (pandas.DataFrame)
    '''
    df = pd.read_csv(path)
    df["name"] = clean(df["name"])
    return df

def clean(name):
    '''
    Only keep alphabets, suffix and prefix and added to represent start and end of name (e.g. _bryant_)
    :params name: names of human beings (pandas.Series)
    :return cleaned dataframe (pandas.Series)
    '''
    def removeSpecialCharacter(s):
        t = "_" 
        for i in s:
            if(i.isalpha()):
                t+=i.lower()
        t += "_"
        return t
    return name.apply(lambda x: removeSpecialCharacter(x))

def get_features(train, val):
    '''
    Get word vectors (ngrams)
    :params train: names of training set (to initialize) (pandas.Series)
    :params validation: names of validation set (pandas.Series)
    :return sparse matrix: [n_samples, n_features] of train and validation
    '''
    preprocess = Preprocessor(train)
    X_train = preprocess.vectorize(train)
    X_val = preprocess.vectorize(val)
    return X_train, X_val

def compute_metrics(pred, labels):
    '''
    Compute accuracy
    :params pred: predictions from logistic regression (list)
    :params labels: ground truth (list)
    :return: accuracy
    '''
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

if __name__ == '__main__':
    '''
    Contains references to (Logistic Regression + N-Grams + Prefix+Suffix):
    What's in a Name? Gender Classification of Names with Character Based Machine Learning Models
    https://arxiv.org/pdf/2102.03692.pdf
    (Perform competitive with other character based deep learning models)
    '''
    # Prepare train and validation set (use for hyperparams tuning)
    df = read(cfg.data.path)
    df_train, df_val = train_test_split(df, test_size=0.33, random_state=1)
    # Prepare features/ vectors (ngrams) for training
    X_train, X_val = get_features(df_train["name"], df_val["name"])
    y_train, y_val = df_train["gender"], df_val["gender"]
    # Hyperparameter optimization if needed
    C, solver = 1.0, 'lbfgs'
    if cfg.optimize.bayes_opt:
        bayes_opt = Optimizer()
        C, solver = bayes_opt.optimize(X_train, X_val, y_train, y_val)
    # Run logistic regression
    clf = LogisticRegression(random_state=0, max_iter=10000, C=C, solver=solver).fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(compute_metrics(y_pred, list(y_val)))

    if cfg.data.test_path:
        df_test = pd.read_csv(cfg.data.test_path)
        # Keep the unprocessed names
        uncleaned_names = df_test["name"]
        # Clean the names
        df_test["name"] = clean(df_test["name"])
        # Prepare features/ vectors (ngrams) for testing
        X_train, X_test = get_features(df_train["name"], df_test["name"])
        # Make predictions
        y_pred = clf.predict(X_test)
        # Export csv
        output = {'name': uncleaned_names, 'gender': y_pred}
        output_df = pd.DataFrame(output)
        output_df.to_csv("./prediction.csv", index=False)
