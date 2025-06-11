from paths import *
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def variance_filtering(X, threshold=0.5):
    selector = VarianceThreshold(threshold=threshold) 
    X_filtered = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)

def co_correlation_filtering(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop)

def feature_correlation(X_train, features, correlation_types):
    correlation_df = pd.DataFrame(index=features)

    if 'slope' in correlation_types:
        slopes = []
        for f in features:
            patient_slopes = []
            for pid, group in X_train.groupby('id'):
                x = group['day'].values.reshape(-1, 1)
                y = group[f].values
                if len(x) > 1:
                    lr = LinearRegression().fit(x, y)
                    patient_slopes.append(lr.coef_[0])
            slopes.append(np.mean(np.abs(patient_slopes)))
        correlation_df['slope'] = pd.Series(slopes, index=features)
        
    if 'target' in correlation_types:
        per_target_corrs = []
        for target in target_columns:
            per_patient_corrs = X_train.groupby('id').apply(lambda g: g[features].corrwith(g[target]))
            corr_mean = per_patient_corrs.mean().abs()
            per_target_corrs.append(corr_mean)
        combined_corr = pd.concat(per_target_corrs, axis=1).mean(axis=1)
        correlation_df['target'] = combined_corr
        
    if 'target_slope' in correlation_types:
        target_slope_corr = []
        for f in features:
            coefs = []
            for pid, g in X_train.groupby('id'):
                if len(g) > 1:
                    x = g['day'].values.reshape(-1, 1)
                    y_feature = g[f].values
                    slopes_t = []
                    for target in target_columns:
                        y_target = g[target].values
                        slope_t = LinearRegression().fit(x, y_target).coef_[0]
                        slopes_t.append(slope_t)
                    slope_f = LinearRegression().fit(x, y_feature).coef_[0]
                    avg_slope_t = np.mean(slopes_t)
                    coefs.append((slope_f, avg_slope_t))
            if coefs:
                corr = np.corrcoef(*zip(*coefs))[0, 1]
            else:
                corr = np.nan
            target_slope_corr.append(corr)
        correlation_df['target_slope'] = pd.Series(target_slope_corr, index=features)
        
    normed_correlation_df = (correlation_df - correlation_df.min()) / (correlation_df.max() - correlation_df.min())
    return normed_correlation_df

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

    ## LASSO per target
    for target in target_columns:
        y_target = y_train[target].values
        lasso = Lasso(alphas=0.05, random_state=42).fit(X_train, y_target)
        non_zero = lasso.coef_ != 0
        features = X_train.columns[non_zero]
        selected_features.update(features)
    X_lasso = X_train[list(selected_features)]

    ## RFE per target
    for target in target_columns:
        estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_lasso, y_train[target])
        features = X_lasso.columns[rfe.support_]
        selected_features.update(features)
    return list(selected_features)


def feature_selection(df, target, correlation_types, test_set_size, correlation_threshold, mi_threshold, cc_threshold, n_features_to_select, step):
    df_path = Path(*df.parts[1:]) ## removes the first part of the path for storage
    print()
    print('Processing', df)
 
    feature_df = pd.read_csv(df)

    ## part 1 removing incomplete features rows (NaN)
    all_indices = set(feature_df.index)
    feature_df_cleaned = feature_df[feature_df.isnull().mean(axis=1) <= 0.05] ## rows with more than 5% NaN are dropped
    indices = set(feature_df_cleaned.index)
    removed_rows = sorted(all_indices - indices)
    print(f"REMOVING NaN ROWS     Removed {len(removed_rows)} rows: {removed_rows}")
    if len(indices) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None
    
    X_all = feature_df_cleaned.iloc[:, 4:]
    train_df, test_df = train_test_split(feature_df_cleaned, test_size=test_set_size, random_state=42)

    scaler = StandardScaler() ## scale the data (z-scores)
    X_train_raw = pd.DataFrame(scaler.fit_transform(train_df.iloc[:, 4:]), columns=X_all.columns, index=train_df.index)
    
    ## part 2 removing incomplete features columns (NaN)
    X_train_0 = X_train_raw.dropna(axis=1)
    col_filter = set(X_train_raw)
    after_col_filter = set(X_train_0.columns)
    removed_cols = sorted(col_filter - after_col_filter)
    print(f"REMOVING NaN COLUMNS     Removed {len(removed_cols)} features: {removed_cols}")
    if len(after_col_filter) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None

    ## part 3 is variance filtering: removing all features that have low variance (do not change over time)
    X_train_0 = variance_filtering(X_train_0, threshold=0.01)
    init_var_filter = after_col_filter
    after_var_filter = set(X_train_0.columns)
    removed_by_variance = sorted(init_var_filter - after_var_filter)
    print(f"VARIANCE FILTERING     Removed {len(removed_by_variance)} features: {removed_by_variance}")
    if len(after_var_filter) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None
    
    ## part 4 is co-correlation filter: removing all features that correlate with each other (are redundant)
    X_train_0 = co_correlation_filtering(X_train_0, threshold=cc_threshold)
    before_cocorr_filter = after_var_filter
    after_cocorr_filter = set(X_train_0.columns)
    removed_by_corr = sorted(before_cocorr_filter - after_cocorr_filter)
    print(f"CO-CORRELATION FILTERING     Removed {len(removed_by_corr)} features: {removed_by_corr}")
    if len(after_cocorr_filter) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None

    ## part 5 is a 3-fold correlation filter: inspects the correlation based on the slope, relation to target and relation to target/slope
    ## first we need to reattach the other columns
    y_train = train_df[target_columns].copy()
    X_train_0['id'] = train_df['id'].values
    X_train_0['day'] = train_df['day'].values
    for target in target_columns:
        X_train_0[target] = train_df[target].values
        features = X_train_0.columns.difference(['id', 'day', target]).tolist()

    correlation_df = feature_correlation(X_train_0, features, correlation_types)
    correlation_df['score'] = (
        correlation_dict['slope']*correlation_df['slope'] + 
        correlation_dict['target']*correlation_df['target'] + 
        correlation_dict['target_slope']*correlation_df['target_slope'])
    
    ## score and save the correlation score tables per dataframe
    correlation_df = correlation_df.sort_values(by='score', ascending=False)
    correlation_df = correlation_df.drop(index=correlation_df.index.intersection(target_columns))
    correlation_df.index.name = 'features'
    correlation_df.to_csv(feature_selection_dir / df_path)
    
    ## remove all features that fall below correlation threshold
    post_correlation_features = correlation_df[correlation_df['score'] >= correlation_threshold].index  ## only features above the threshold continue
    X_train_1 = X_train_0[post_correlation_features]
    
    before_corr_score = set(after_cocorr_filter)
    after_corr_score = set(post_correlation_features)
    removed_by_corr_score = sorted(before_corr_score - after_corr_score)
    print(f"CORRELATION SCORE THRESHOLDING     Removed {len(removed_by_corr_score)} features: {removed_by_corr_score}")
    if len(post_correlation_features) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None
    
    ## part 6 is mutual information filtering
    post_mi_features = mutual_information_filtering(X_train_1, y_train, mi_threshold)
    X_train_2 = X_train_1[post_mi_features]
    
    before_mi = set(post_correlation_features)
    after_mi = set(post_mi_features)
    removed_by_mi = sorted(before_mi - after_mi)
    print(f"MUTUAL INFORMATION FILTERING     Removed {len(removed_by_mi)} features: {removed_by_mi}")
    if len(post_mi_features) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None
    if len(post_mi_features) < 10:
        print("Less than 10 features is not enough for LASSO")

    ## part 7 is LASSO and RFE feature selection 
    selected_features = lasso_rfe(X_train_2, y_train, n_features_to_select)
    before_rfe = set(post_mi_features)
    after_rfe = set(selected_features)
    removed_by_rfe = sorted(before_rfe - after_rfe)
    print(f"LASSO & RFE     Removed {len(removed_by_rfe)} features: {removed_by_rfe}")
    print(f"FINAL FEATURE COUNT     Selected {len(selected_features)} features: {selected_features}")
    if len(after_rfe) < 2:
        print("All features were deminished around this step...")
        return None, None, None, None, None
    
    X_train = X_train_0[selected_features]    
    y_test = test_df[target]
    X_test = test_df[selected_features]
    
    return X_train, y_train, X_test, y_test, selected_features


def model_training(X_train, y_train, X_test, y_test, selected_features, model_type='rf'):
    model_name = f"{model_type}_{exercise}_{target}" ## the naming convention needs to be determined
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train[selected_features], y_train)
        joblib.dump(model, f'models/{model_name}.pkl') 

        y_pred = model.predict(X_test[selected_features])
    
    elif model_type == 'svm':
        model = SVR(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{model_name}.pkl')
        
        y_pred = model.predict(X_test)
        
    elif model_type == 'gbm':
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',  # Change this to 'multiclass' if you have more than two classes
            'metric': 'binary_error',  # Change this to 'multi_logloss' for multiclass
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
        }

        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, early_stopping_rounds=100)
        joblib.dump(model, f'models/{model_name}.pkl')

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred > 0.5).astype(int)    
        
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
            
    return accuracy, class_report


if __name__ == '__main__':
    pe_files = [f for f in (df_features_dir / pe).iterdir() if f.is_file()]
    pp_files = [f for f in (df_features_dir / pp).iterdir() if f.is_file()] 
    df_list = pe_files + pp_files
    
    df_list = [path for path in df_list if 'MPT' not in str(path)] ## excludes MPT feature dataframes
    
    correlation_dict = {
        'slope': 0.2,
        'target': 0.25,
        'target_slope': 0.3
    }

    correlation_types = list(correlation_dict.keys())

    for df, target in list(product(df_list, target_columns)):
        X_train, y_train, X_test, y_test, selected_features = feature_selection(
            df, target, correlation_types,
            test_set_size=0.15, 
            correlation_threshold=0.2, ## 0.2 
            mi_threshold=0.05, ## 0.05
            cc_threshold=0.75, ## 0.75
            n_features_to_select=10, 
            step=1,
        )
        
        # accuracy, class_report = model_training(X_train, y_train, X_test, y_test, selected_features, model_type='rf')
        
        

 