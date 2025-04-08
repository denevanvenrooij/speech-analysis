import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def feature_correlation(X_scaled, features, correlation_types):
    
    correlation_df = pd.DataFrame(index=features)

    if 'variance' in correlation_types:
        variance_corr = X_scaled.groupby('id')[features].std().mean()
        correlation_df['variance'] = variance_corr

    if 'slope' in correlation_types: 
        slopes = []
        for f in features:
            patient_slopes = []
            for pid, group in X_scaled.groupby('id'):
                x = group['time_norm'].values.reshape(-1, 1)
                y = group[f].values
                if len(x) > 1:
                    lr = LinearRegression().fit(x, y)
                    patient_slopes.append(lr.coef_[0])
            slopes.append(np.mean(np.abs(patient_slopes)))
        correlation_df['slope'] = pd.Series(slopes, index=features)
        
    if 'target' in correlation_types:
        per_patient_corrs = X_scaled.groupby('id').apply(lambda g: g[features].corrwith(g['target']))
        corr_mean = per_patient_corrs.mean().abs()
        correlation_df['target'] = corr_mean
        
    if 'target_slope' in correlation_types:
        target_slope_corr = []
        for f in features:
            coefs = []
            for pid, g in X_scaled.groupby('id'):
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


def LASSO():

    lasso = LassoCV(cv=5, random_state=42).fit(X_uncorrelated, y_targets.values.ravel())
    lasso_selected = X_uncorrelated.columns[(lasso.coef_ != 0)]

    rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42) ## recursive feature elimination
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=30, step=50)
    rfe.fit(X_uncorrelated[lasso_selected], y_targets)
    selected_features = X_uncorrelated[lasso_selected].columns[rfe.support_]

    return selected_features


if __name__=='__main__':
    
    exercises = ['MPT','SEN','SPN','VOW']
    
    correlation_types = [
        'variance',
        'slope',
        'target',
        'target_slope',
    ]
    
    variance_weight = 0.15
    slope_weight = 0.2
    target_weight = 0.25
    target_slope_weight = 0.3
    
    filter_threshold = 0.5
    
    for exercise in exercises:
        feature_df_path = f'feature_dataframes/all_features_{exercise}.csv'
        feature_df = pd.read_csv(feature_df_path)
        
        X_features = feature_df[:,:-3] ## change into all but the last columns
        y_targets = feature_df[['target_1', 'target_2', 'target_3']] ## select the last three columns or targets
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        features = X_scaled[1:500] ## insert here which columns are features
        
        correlation_df = feature_correlation(X_scaled, features, correlation_types)
        
        correlation_df['score'] = target_weight*correlation_df["target_slope"] + target_slope_weight*correlation_df["target"] + slope_weight*correlation_df["slope"] + variance_weight*correlation_df["variance"]
        correlation_df = correlation_df.sort_values(by='score', ascending=False)
        print(correlation_df.head(100))

        correlation_df.to_csv(f'dataframes/correlation_features_{exercise}.csv')
        
        filtered_df = correlation_df[correlation_df['score'] >= filter_threshold]
        filtered_features = filtered_df.index
        
        X_uncorrelated = pd.DataFrame(X_scaled, columns=X_features.columns)
        
        lasso = LassoCV(cv=5, random_state=42).fit(X_uncorrelated[filtered_features], y_targets.values.ravel())
        lasso_selected = X_uncorrelated.columns[(lasso.coef_ != 0)]
        
        rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=30, step=50)
        rfe.fit(X_uncorrelated[lasso_selected], y_targets)
    
        selected_features = X_uncorrelated[lasso_selected].columns[rfe.support_]
        
        print(f"Selected features for {exercise} after LASSO and RFE: {selected_features}")
