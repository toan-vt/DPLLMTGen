import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import time
from joblib import Parallel, delayed
import itertools

def to_numbers_drop_wrong(df_test_, df_synthetic_, n_bins, categorical_only=False):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    categorical_columns = df_test.select_dtypes(include=['object']).columns
    numerical_columns = df_test.select_dtypes(include=['int64', 'float64']).columns

    # convert categorical values to numbers
    for col in categorical_columns:
        mapping_dict = {}
        unique_values = list(df_test[col].unique())
        unique_values = [str(s) for s in unique_values]
        unique_values = list(set(unique_values))
        for idx, value in enumerate(sorted(unique_values)):
            mapping_dict[value] = idx
        df_synthetic[col] = df_synthetic[col].map(mapping_dict)
        df_test[col] = df_test[col].map(mapping_dict)

    # reorder columns
    df_synthetic = df_synthetic[df_test.columns]

    if not categorical_only:
        for col in numerical_columns:
            # if len(df_test[col].unique()) <= n_bins:
            #     continue
            if df_test[col].dtype == 'int64' and df_synthetic[col].dtype == 'float64':
                df_synthetic[col] = df_synthetic[col].astype(int)
            lower, higher = df_test[col].min()-0.000001, df_test[col].max()+0.000001
            edges = np.linspace(lower, higher, n_bins)
            df_synthetic[col] = np.digitize(df_synthetic[col], edges)
            df_test[col] = np.digitize(df_test[col], edges)

    return df_test, df_synthetic


def to_numbers(df_test_, df_synthetic_, n_bins, categorical_only=False):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    categorical_columns = df_test.select_dtypes(include=['object']).columns
    numerical_columns = df_test.select_dtypes(include=['int64', 'float64']).columns

    # convert categorical values to numbers
    for col in categorical_columns:
        mapping_dict = {}
        unique_values = list(df_test[col].unique()) + list(df_synthetic[col].unique())
        unique_values = [str(s) for s in unique_values]
        unique_values = list(set(unique_values))
        for idx, value in enumerate(sorted(unique_values)):
            mapping_dict[value] = idx
        df_synthetic[col] = df_synthetic[col].map(mapping_dict)
        df_test[col] = df_test[col].map(mapping_dict)

    # reorder columns
    df_synthetic = df_synthetic[df_test.columns]

    if not categorical_only:
        for col in numerical_columns:
            # if len(df_test[col].unique()) <= n_bins:
            #     continue
            if df_test[col].dtype == 'int64' and df_synthetic[col].dtype == 'float64':
                df_synthetic[col] = df_synthetic[col].astype(int)
            lower, higher = df_test[col].min()-0.000001, df_test[col].max()+0.000001
            edges = np.linspace(lower, higher, n_bins)
            df_synthetic[col] = np.digitize(df_synthetic[col], edges)
            df_test[col] = np.digitize(df_test[col], edges)

    return df_test, df_synthetic

def to_2_margin(df_test_, df_synthetic_, selected_cols=None):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    df_synthetic_2 = pd.DataFrame()
    df_test_2 = pd.DataFrame()
    if selected_cols is None:
        selected_cols = df_test.columns
    for col_1 in selected_cols:
        for col_2 in selected_cols:
            if col_1 == col_2:
                continue
            df_synthetic_2[f"{col_1}_{col_2}"] = df_synthetic[col_1].astype(str) + "_" + df_synthetic[col_2].astype(str)
            df_test_2[f"{col_1}_{col_2}"] = df_test[col_1].astype(str) + "_" + df_test[col_2].astype(str)
    return df_test_2, df_synthetic_2
    
def process_3_columns(df_test, df_synthetic, col_1, col_2, col_3):
    if col_1 == col_2 or col_1 == col_3 or col_2 == col_3:
        return None, None
    
    syn_col = df_synthetic[col_1].astype(str) + "_" + df_synthetic[col_2].astype(str) + "_" + df_synthetic[col_3].astype(str)
    test_col = df_test[col_1].astype(str) + "_" + df_test[col_2].astype(str) + "_" + df_test[col_3].astype(str)
    return syn_col, test_col

def to_3_margin(df_test_, df_synthetic_, selected_cols=None):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    df_synthetic_3 = pd.DataFrame()
    df_test_3 = pd.DataFrame()
    if selected_cols is None:
        selected_cols = df_test.columns
    combinations = list(itertools.product(selected_cols, repeat=3))
    results = Parallel(n_jobs=-1)(delayed(process_3_columns)(df_test, df_synthetic, *combination) for combination in combinations)
    for idx, (syn_col, test_col) in enumerate(results):
        if syn_col is None:
            continue
        df_synthetic_3[f"{combinations[idx][0]}_{combinations[idx][1]}_{combinations[idx][2]}"] = syn_col
        df_test_3[f"{combinations[idx][0]}_{combinations[idx][1]}_{combinations[idx][2]}"] = test_col

    return df_test_3, df_synthetic_3

def process_k_columns(df_test, df_synthetic, cols):
    # check duplicate columns
    if len(cols) != len(set(cols)):
        return None, None, None
    col_name = "_".join(cols)
    syn_col = df_synthetic[cols[0]].astype(str)
    test_col = df_test[cols[0]].astype(str)
    for col in cols[1:]:
        syn_col = syn_col + "_" + df_synthetic[col].astype(str)
        test_col = test_col + "_" + df_test[col].astype(str)
    return  col_name, syn_col, test_col

def to_k_margin(df_test_, df_synthetic_, k, selected_cols=None):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    df_synthetic_k = pd.DataFrame()
    df_test_k = pd.DataFrame()
    if selected_cols is None:
        selected_cols = df_test.columns
    combinations = list(itertools.combinations(selected_cols, k))
    results = Parallel(n_jobs=-1)(delayed(process_k_columns)(df_test, df_synthetic, combination) for combination in combinations)
    for col_name, syn_col, test_col in results:
        if syn_col is None:
            continue
        df_synthetic_k[col_name] = syn_col
        df_test_k[col_name] = test_col

    return df_test_k, df_synthetic_k

def get_tvd(df_test_, df_synthetic_, selected_cols=None):
    df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
    tvds = []
    cols = []
    if selected_cols is None:
        selected_cols = df_test.columns
    for col in selected_cols:
        test_prob_ = dict(df_test[col].value_counts(normalize=True))
        synthetic_prob_ = dict(df_synthetic[col].value_counts(normalize=True))
        # covert all keys to string
        test_prob = {str(key): value for key, value in test_prob_.items()}
        synthetic_prob = {str(key): value for key, value in synthetic_prob_.items()}
        for key in test_prob:
            if key not in synthetic_prob:
                synthetic_prob[key] = 0
        for key in synthetic_prob:
            if key not in test_prob:
                test_prob[key] = 0
                # del synthetic_prob[key]

        test_prob = dict(sorted(test_prob.items()))
        synthetic_prob = dict(sorted(synthetic_prob.items()))
        test_prob = list(test_prob.values())
        synthetic_prob = list(synthetic_prob.values())

        tvd = sum([abs(test_prob[i] - synthetic_prob[i]) for i in range(len(test_prob))]) / 2
        tvds.append(tvd)
        cols.append(col)
    
    return pd.DataFrame({'column': cols, 'tvd': tvds})

def get_xgboost_performance(df_test_, df_synthetic_, others=True, fairness_metrics=False):
    # try:
        df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
        df_test, df_synthetic = to_numbers(df_test, df_synthetic, 20, categorical_only=True)
        df_test = df_test.dropna()
        df_synthetic = df_synthetic.dropna()
        target_col = df_test.columns[-1]

        X_train, y_train = df_synthetic.drop(columns=[target_col]), df_synthetic[target_col]
        X_test, y_test = df_test.drop(columns=[target_col]), df_test[target_col]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #train a xgboost with grid search
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier
        # xgb = XGBClassifier()
        # parameters = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [3, 5, 10, 20],
        #     'learning_rate': [0.01, 0.05, 0.1]
        # }
        # clf = GridSearchCV(xgb, parameters, cv=5, n_jobs=-1, verbose=False)
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        # get accuracy
        acc = clf.score(X_test, y_test)
        # get auc
        if not others:
            return acc
        else:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
            y_pred = clf.predict(X_test)
            print("xgb: ", sum(y_pred), len(y_pred))
            prec = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            if fairness_metrics: #TODO hard code for adult dataset
                female_pos_prob = len(df_synthetic[(df_synthetic["sex"] == 0) & (df_synthetic["income"] == 1)]) / len(df_synthetic[df_synthetic["sex"] == 0])
                male_pos_prob = len(df_synthetic[(df_synthetic["sex"] == 1) & (df_synthetic["income"] == 1)]) / len(df_synthetic[df_synthetic["sex"] == 1])
                train_data_dp_diff = male_pos_prob - female_pos_prob
                from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
                sensitive_features = df_test["sex"].values
                dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
                eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
                return acc, prec, recall, f1, auc, train_data_dp_diff, dp_diff, eo_diff
            
            return acc, prec, recall, f1, auc
    # except:
    #     return None, None, None, None, None

def get_lr_performance(df_test_, df_synthetic_, others=True):
    try:
        df_test, df_synthetic = df_test_.copy(), df_synthetic_.copy()
        df_test, df_synthetic = to_numbers(df_test, df_synthetic, 20, categorical_only=True)
        df_test = df_test.dropna()
        df_synthetic = df_synthetic.dropna()
        target_col = df_test.columns[-1]

        X_train, y_train = df_synthetic.drop(columns=[target_col]), df_synthetic[target_col]
        X_test, y_test = df_test.drop(columns=[target_col]), df_test[target_col]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #train a linear regression with grid search
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        # lr = LogisticRegression()
        # parameters = {
        #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #     'penalty': ['l1', 'l2']
        # }
        # clf = GridSearchCV(lr, parameters, cv=5, n_jobs=-1, verbose=False)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # get accuracy
        acc = clf.score(X_test, y_test)
        # get auc
        if not others:
            return acc
        else:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
            y_pred = clf.predict(X_test)
            print("lr: ", sum(y_pred), len(y_pred))
            prec = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            return acc, prec, recall, f1, auc
    except:
        return None, None, None, None, None
    # return acc