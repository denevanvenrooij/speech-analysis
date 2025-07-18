from paths import *
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.model_selection import KFold

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
from datetime import datetime

time = datetime.now().strftime('%y%m%d_%H%M%S')
log_filename = f"logs/5_mf_{time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

def fill_nan(df, group_by='id'):
    df_filled = df.copy()
    feature_cols = df.select_dtypes(include='number').columns.difference(['id', 'mic', 'exercise', 'day', 'vowel'])
    group_means = df.groupby(group_by)[feature_cols].transform(lambda x: x.fillna(x.mean()))
    global_means = df[feature_cols].mean()
    df_filled[feature_cols] = group_means.fillna(global_means)
    return df_filled

def drop_nan(feature_df):
    data_columns = feature_df.columns[5:]
    data_rows = feature_df.index[1:]
    while feature_df.loc[:, data_columns].isna().any().any():
        row_nan_ratio = feature_df.loc[:, data_columns].isna().sum(axis=1) / len(data_columns)
        col_nan_ratio = feature_df.loc[data_rows, data_columns].isna().sum(axis=0) / len(data_rows)
        worst_row = row_nan_ratio.idxmax()
        worst_col = col_nan_ratio.idxmax()
        if row_nan_ratio.max() >= col_nan_ratio.max():
            worst_row = row_nan_ratio.idxmax()
            feature_df = feature_df.drop(index=worst_row)
        else:
            worst_col = col_nan_ratio.idxmax()
            feature_df = feature_df.drop(columns=worst_col)
        data_columns = feature_df.columns[4:]
        data_rows = feature_df.index[1:]
    return feature_df

def co_correlation_filtering(X_train, cc_threshold):
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cc_threshold)]
    return X_train.drop(columns=to_drop)

def mutual_information_filtering(X_train, y_train, mi_threshold):
    per_target_mi = []
    for target in target_columns:
        y_train_per_target = y_train[target]
        mi_scores = mutual_info_regression(X_train, y_train_per_target)
        mi_series = pd.Series(mi_scores, index=X_train.columns)
        per_target_mi.append(mi_series)
    mi_df = pd.concat(per_target_mi, axis=1)
    mean_mi = mi_df.mean(axis=1)
    mi_selected_features = mean_mi[mean_mi > mi_threshold].index
    return mi_selected_features

def lasso_rfe(X_train, y_train, n_features_to_select):
    selected_features = set()
    for target in target_columns:
        y_target = y_train[target].values
        lasso = Lasso(alpha=0.05, random_state=42).fit(X_train, y_target)
        non_zero = lasso.coef_ != 0
        features = X_train.columns[non_zero]
        selected_features.update(features)
    X_lasso = X_train[list(selected_features)]
    for target in target_columns:
        estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_lasso, y_train[target])
        features = X_lasso.columns[rfe.support_]
        selected_features.update(features)
    return list(selected_features)

def feature_selection(train_df, mi_threshold, cc_threshold, n_features_to_select):
    
    # ## part 1 removing empty rows/column
    # all_rows = set(feature_df.index)
    # all_cols = set(feature_df)
    
    # cleaned_feature_df = fill_nan(feature_df, group_by='id')
    
    # cleaned_rows = set(cleaned_feature_df.index)
    # cleaned_cols = set(cleaned_feature_df)
    # removed_rows = sorted(all_rows - cleaned_rows)
    # removed_cols = sorted(all_cols - cleaned_cols)
    # logging.info(f"REMOVING NaN ROWS     Removed {len(removed_rows)} rows: {removed_rows}")
    # if len(cleaned_rows) < 2:
    #     logging.info("All features were deminished at this step...")
    #     return None, None, None, None, None
    # logging.info(f"REMOVING NaN COLUMNS     Removed {len(removed_cols)} features: {removed_cols}")
    # if len(cleaned_cols) < 2:
    #     logging.info("All features were deminished at this step...")
    #     return None, None, None, None, None
    
    feature_start_idx = train_df.columns.get_loc('PP_F0')
    X_train = train_df.iloc[:, feature_start_idx:]
    y_train = train_df[target_columns].copy()

    ## part 3 is co-correlation filter: removing all features that correlate with each other (are redundant)
    X_train_1 = co_correlation_filtering(X_train, cc_threshold)
    before_cocorr_filter = set(X_train)
    after_cocorr_filter = set(X_train_1.columns)
    removed_by_corr = sorted(before_cocorr_filter - after_cocorr_filter)
    logging.info(f"CO-CORRELATION FILTERING     Removed {len(removed_by_corr)} features: {removed_by_corr}")
    if len(after_cocorr_filter) < 2:
        logging.info("All features were deminished at this step...")
        return None, None, None
    
    ## part 3 is mutual information filtering
    post_mi_features = mutual_information_filtering(X_train_1, y_train, mi_threshold)
    X_train_2 = X_train_1[post_mi_features]
    
    before_mi = set(after_cocorr_filter)
    after_mi = set(post_mi_features)
    removed_by_mi = sorted(before_mi - after_mi)
    logging.info(f"MUTUAL INFORMATION FILTERING     Removed {len(removed_by_mi)} features: {removed_by_mi}")
    if len(post_mi_features) < 2:
        logging.info("All features were deminished at this step...")
        return None, None, None
    if len(post_mi_features) < 25:
        logging.info("Less than 25 features is not enough for LASSO")

    ## part 4 is LASSO and RFE feature selection 
    selected_features = lasso_rfe(X_train_2, y_train, n_features_to_select)
    before_rfe = set(post_mi_features)
    after_rfe = set(selected_features)
    removed_by_rfe = sorted(before_rfe - after_rfe)
    logging.info(f"LASSO & RFE     Removed {len(removed_by_rfe)} features: {removed_by_rfe}")
    logging.info(f"FINAL FEATURE COUNT     Selected {len(selected_features)} features: {selected_features}")
    if len(after_rfe) < 2:
        logging.info("All features were deminished at this step...")
        return None, None, None
    
    X_train = X_train_2[selected_features]    
    
    return X_train, y_train, selected_features


def model_training(X_train, y_train, X_test, y_test, selected_features, model_path):
    logging.info(f"Creating a model for {model_path}")

    base_model = RandomForestRegressor(n_estimators=200, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train[selected_features], y_train)
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test[selected_features])

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    logging.info(f"Mean Squared Errors per target: {mse}")
    logging.info(f"R² Scores per target: {r2}")

    return mse, r2


def cv_split_pipeline(df, mi_threshold, cc_threshold, n_features_to_select):
    scores = []

    ## part 1 is filling NaN cells with average values
    print(df.shape)
    print(df.isna().sum().sum())
    cleaned_feature_df = fill_nan(df, group_by='id')
    print(cleaned_feature_df.shape)
    print(cleaned_feature_df.isna().sum().sum())
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(cv.split(cleaned_feature_df)):
        train_df = cleaned_feature_df.iloc[train_index].reset_index(drop=True)
        test_df = cleaned_feature_df.iloc[test_index].reset_index(drop=True)

        X_train, y_train, selected_features = feature_selection(train_df.copy(), mi_threshold=mi_threshold, cc_threshold=cc_threshold, n_features_to_select=n_features_to_select)

        scaler = StandardScaler().fit(train_df[selected_features])
        X_train = pd.DataFrame(scaler.transform(train_df[selected_features]), columns=selected_features, index=train_df.index)
        X_test = pd.DataFrame(scaler.transform(test_df[selected_features]), columns=selected_features, index=test_df.index)
        y_train = train_df[target_columns]
        y_test = test_df[target_columns]

        mse, r2 = model_training(X_train, y_train, X_test, y_test, selected_features, models_dir / f"rf_{df_path.stem}_fold_{fold}.pkl")
        scores.append((mse, r2))
    return scores


if __name__ == '__main__':
    pe_files = [f for f in (df_features_dir / pe).iterdir() if f.is_file()]
    pp_files = [f for f in (df_features_dir / pp).iterdir() if f.is_file()] 
    df_list = pe_files + pp_files
    df_list = [path for path in df_list if 'MPT' not in str(path)] ## excludes MPT feature dataframes

    mi_threshold = 0.01
    cc_threshold = 0.99
    n_features_to_select = 25

    for df_path in df_list:
        logging.info(f"Now running {df_path} with")
        logging.info(f"Mutual information threshold: {mi_threshold}")
        logging.info(f"Co-correlation threshold: {cc_threshold}")
        logging.info(f"Features to select: {n_features_to_select}")
        
        df = pd.read_csv(df_path)

        scores = cv_split_pipeline(df=df, mi_threshold=mi_threshold, cc_threshold=cc_threshold, n_features_to_select=n_features_to_select)

        for i, (mse, r2) in enumerate(scores, 1):
            logging.info(f"Fold {i}: Mean squared error = {mse}, R2 = {r2}") 