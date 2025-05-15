from paths import *
import pandas as pd
import numpy as np
from itertools import product
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.model_selection import train_test_split


def feature_correlation(X_train, features, correlation_types):
    correlation_df = pd.DataFrame(index=features)

    if 'variance' in correlation_types:
        variance_corr = X_train.groupby('id')[features].std().mean()
        correlation_df['variance'] = variance_corr

    if 'slope' in correlation_types:
        slopes = []
        for f in features:
            patient_slopes = []
            for pid, group in X_train.groupby('id'):
                x = group['time_norm'].values.reshape(-1, 1)
                y = group[f].values
                if len(x) > 1:
                    lr = LinearRegression().fit(x, y)
                    patient_slopes.append(lr.coef_[0])
            slopes.append(np.mean(np.abs(patient_slopes)))
        correlation_df['slope'] = pd.Series(slopes, index=features)

    if 'target' in correlation_types:
        per_patient_corrs = X_train.groupby('id').apply(lambda g: g[features].corrwith(g['target']))
        corr_mean = per_patient_corrs.mean().abs()
        correlation_df['target'] = corr_mean

    if 'target_slope' in correlation_types:
        target_slope_corr = []
        for f in features:
            coefs = []
            for pid, g in X_train.groupby('id'):
                if len(g) > 1:
                    x = g['time_norm'].values.reshape(-1, 1)
                    y_feature = g[f].values
                    y_target = g['target'].values
                    slope_f = LinearRegression().fit(x, y_feature).coef_[0]
                    slope_t = LinearRegression().fit(x, y_target).coef_[0]
                    coefs.append((slope_f, slope_t))
            if coefs:
                corr = np.corrcoef(*zip(*coefs))[0, 1]
            else:
                corr = np.nan
            target_slope_corr.append(corr)
        correlation_df['target_slope'] = pd.Series(target_slope_corr, index=features)

    normed_correlation_df = (correlation_df - correlation_df.min()) / (correlation_df.max() - correlation_df.min())

    return normed_correlation_df


def mutual_information_filtering(X_train, y_train, mi_threshold=0.01):
    mi_scores = mutual_info_regression(X_train, y_train)
    mi_series = pd.Series(mi_scores, index=X_train.columns)
    
    mi_selected_features = mi_series[mi_series > mi_threshold].index
    
    return mi_selected_features


def lasso_rfe(X_train, y_train, features, n_estimators=100, n_features_to_select=30, step=1):
    lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    lasso_selected = features[lasso.coef_ != 0] ## what is inserted is an index object, make sure this does not cause problems

    rfe_estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X_train[lasso_selected], y_train)

    selected_features = X_train[lasso_selected].columns[rfe.support_]
    
    return selected_features


def feature_selection(exercise, mic, target, weights, correlation_types, test_set_size, correlation_threshold, mi_threshold, n_estimators, n_features_to_select, step):
    feature_df = pd.read_csv(f'dataframes_features/all_features_{exercise}_{mic}.csv')
    X_all = feature_df.iloc[:, :-1]

    train_df, test_df = train_test_split(feature_df, test_size=test_set_size, random_state=42)

    scaler = StandardScaler()
    X_train_1 = pd.DataFrame(scaler.fit_transform(train_df.iloc[:, :-3]), columns=X_all.columns, index=train_df.index)
    X_train_1['id'] = train_df['id'].values ## reattach the non-feature columns to X_train_1
    X_train_1['day'] = train_df['day'].values
    X_train_1[target] = train_df[target].values
    y_train = train_df[target]
    
    features = X_all.columns.tolist()
    correlation_df = feature_correlation(X_train_1, features, correlation_types)

    correlation_df['score'] = (
        weights['variance']*correlation_df['variance'] + weights['slope']*correlation_df['slope'] +
        weights['target']*correlation_df['target'] + weights['target_slope']*correlation_df['target_slope'])

    correlation_df = correlation_df.sort_values(by='score', ascending=False) ## ordering and saving the correlation scores dataframe
    correlation_df.to_csv(f'dataframes_features/correlation_features_{exercise}_{target}.csv')

    post_correlation_features = correlation_df[correlation_df['score'] >= correlation_threshold].index  ## only features above the threshold continue
    X_train_2 = X_train_1[post_correlation_features]

    post_mi_features = mutual_information_filtering(X_train_2, y_train, mi_threshold)
    X_train_3 = X_train_1[post_mi_features]
    
    selected_features = lasso_rfe(X_train_3, y_train, post_mi_features, n_estimators, n_features_to_select, step)
    X_train = X_train_1[selected_features]
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
    
    targets = {'target_bnp', 'target_bw'}
    correlation_types = [
        'variance', 
        'slope', 
        'target', 
        'target_slope'
    ]
    weights = {
        'variance': 0.15,
        'slope': 0.2,
        'target': 0.25,
        'target_slope': 0.3
    }

    for exercise, target in list(product(exercises, targets)):
        X_train, y_train, X_test, y_test, selected_features = feature_selection(
            exercise, target, weights, correlation_types,
            test_set_size=0.15, 
            correlation_threshold=0.5, 
            mi_threshold=0.01, 
            n_estimators=100, 
            n_features_to_select=25, 
            step=1,
        )
        
        accuracy, class_report = model_training(X_train, y_train, X_test, y_test, selected_features, model_type='rf')
        
        

 