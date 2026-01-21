from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import os
import joblib

def load_noc_mapping(data_dir):
    """
    Creates a mapping from NOC Code (e.g. USA) to Country Name (e.g. United States)
    using summerOly_athletes.csv
    """
    athletes_path = os.path.join(data_dir, 'summerOly_athletes.csv')
    if not os.path.exists(athletes_path):
        print("Warning: athletes file not found for mapping.")
        return {}
    
    df = pd.read_csv(athletes_path)
    # Get most frequent Team name for each NOC code
    # We want a map: NOC Data (Code) -> Medal Data (Name)
    # But Medal Data "NOC" is Name. 
    # Check consistency: Medal Data "United States". Athletes "United States" -> NOC "USA".
    
    # We will try to map Code -> Name
    mapping = df.groupby('NOC')['Team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).to_dict()
    
    # Manual fixes for common discrepancies if needed
    # (Based on standard datasets)
    mapping['USA'] = 'United States'
    mapping['GBR'] = 'Great Britain'
    mapping['CHN'] = 'China'
    mapping['FRA'] = 'France'
    mapping['URS'] = 'Soviet Union'
    mapping['GDR'] = 'East Germany'
    mapping['FRG'] = 'West Germany'
    mapping['RUS'] = 'Russia'
    mapping['ROU'] = 'Romania' # check spelling
    
    return mapping

def main():
    print("Starting XGBoost + Bootstrap Prediction Pipeline...")
    
    # 1. Load Data
    processed_dir = 'processed_data'
    data_dir = 'Data'
    if not os.path.exists(data_dir): 
        data_dir = '../Data'
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} not found. Please run PCA.py first.")
        return

    # Load PCA Scores (NOC latent features from 2020)
    # Shape: N_countries x 5
    pca_scores_path = os.path.join(processed_dir, 'pca_scores_matrix.csv')
    lstm_features_path = os.path.join(processed_dir, 'lstm_generated_features.csv') # New input
    
    if not os.path.exists(pca_scores_path) or not os.path.exists(lstm_features_path):
        print("PCA scores or LSTM features not found. Please run PCA.py and LSTM.py.")
        return
        
    df_pca_scores = pd.read_csv(pca_scores_path)
    # PCA output usually has NOC Code in index or unnamed column
    if 'Unnamed: 0' in df_pca_scores.columns:
        df_pca_scores = df_pca_scores.rename(columns={'Unnamed: 0': 'NOC_Code'})
    elif 'NOC' in df_pca_scores.columns:
         df_pca_scores = df_pca_scores.rename(columns={'NOC': 'NOC_Code'})
         
    df_lstm_features = pd.read_csv(lstm_features_path)
    # LSTM features 'NOC' is Country Name
    
    print(f"Loaded PCA Scores: {df_pca_scores.shape}")
    print(f"Loaded LSTM Features: {df_lstm_features.shape}")
    
    # --- Fix NOC Mismatch ---
    # df_pca_scores has Codes (USA), df_lstm_features has Names (United States)
    # We need to add 'NOC_Name' to df_pca_scores 
    
    noc_map = load_noc_mapping(data_dir)
    print(f"Loaded NOC mapping for {len(noc_map)} codes.")
    
    df_pca_scores['NOC'] = df_pca_scores['NOC_Code'].map(noc_map)
    
    # Fill missing or identity if map fails (some codes might just be names?)
    df_pca_scores['NOC'] = df_pca_scores['NOC'].fillna(df_pca_scores['NOC_Code'])
    
    # Check overlap
    pca_names = set(df_pca_scores['NOC'].unique())
    lstm_names = set(df_lstm_features['NOC'].unique())
    overlap = pca_names.intersection(lstm_names)
    print(f"NOC Overlap: {len(overlap)} countries.")
    
    if len(overlap) < 5:
        print("Critical Warning: Very low overlap between PCA and LSTM data. Check NOC naming.")
        print("Sample PCA Names:", list(pca_names)[:5])
        print("Sample LSTM Names:", list(lstm_names)[:5])
    
    # Check for expected columns in LSTM features
    expected_cols = ['Is_Host', 'Alpha', 'LSTM_Gold_Trend', 'LSTM_Total_Trend']
    for col in expected_cols:
        if col not in df_lstm_features.columns:
            print(f"Warning: {col} missing in LSTM features.")
            df_lstm_features[col] = 0.0
    
    # Load raw data for Ground Truth (Target y)
    df_medals = pd.read_csv(os.path.join(data_dir, 'summerOly_medal_counts.csv'))
    
    # 2. Merge Data
    # Base: df_lstm_features (contains NOC, Year, Trends, Vectors)
    # Join PCA (on NOC)
    # We join on 'NOC' (Name)
    
    df_merged = pd.merge(df_lstm_features, df_pca_scores, on='NOC', how='inner')
    
    # Join Actual Medals (Target)
    # Join on Name and Year
    df_merged = pd.merge(df_merged, df_medals[['NOC', 'Year', 'Gold', 'Total']], on=['NOC', 'Year'], how='left', suffixes=('_feat', '_target'))
    
    # Use 'Gold_feat' (from LSTM input vector) as ground truth if available and aligns with Year.
    # Note: LSTM output 'Gold' is the actual gold for that year (item['feat_gold']).
    # So we can use it.
    
    # Ensure we use the correct target column
    # For training (Year < 2028), we use the actual values.
    # 'Gold_feat' columns are from LSTM generation which passed through real values.
    
    # Rename for clarity
    if 'Gold_feat' in df_merged.columns:
        df_merged['Gold_Actual'] = df_merged['Gold_feat']
        df_merged['Total_Actual'] = df_merged['Total_feat']
    elif 'Gold' in df_merged.columns: # If suffix wasn't added
        df_merged['Gold_Actual'] = df_merged['Gold']
        df_merged['Total_Actual'] = df_merged['Total']
        
    print(f"Merged Dataset Shape: {df_merged.shape}")
    
    # Save Integrated Dataset
    integrated_path = os.path.join(processed_dir, 'xgboost_integrated_dataset.csv')
    df_merged.to_csv(integrated_path, index=False)
    print(f"Integrated dataset saved to {integrated_path}")
    
    # 3. Model Training & Prediction
    
    # FEATURES for XGBoost (Equation 7)
    feature_cols = ['LSTM_Gold_Trend', 'LSTM_Total_Trend', 'Is_Host', 'Alpha'] + [c for c in df_pca_scores.columns if 'PC' in c]
    
    # Define Tasks
    tasks = [
        {'target': 'Gold_Actual', 'col_name': 'Gold'},
        {'target': 'Total_Actual', 'col_name': 'Total'}
    ]
    
    # For prediction, we need the row where Year == 2028
    df_2028 = df_merged[df_merged['Year'] == 2028].copy()
    
    if df_2028.empty:
        print("Error: No 2028 prediction rows found in dataset.")
        return
        
    results_2028 = df_2028[['NOC']].copy()
    
    for task in tasks:
        target_col = task['target']
        result_name = task['col_name']
        
        print(f"\nTraining XGBoost for {result_name} Medals...")
        
        # Train Data: Valid targets only (Year < 2028)
        df_train = df_merged.dropna(subset=[target_col])
        df_train = df_train[df_train['Year'] < 2028]
        
        if df_train.empty:
            print(f"Skipping {result_name}: No training targets found.")
            continue
            
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        
        print(f"  Training samples: {len(X_train)}")
        
        X_pred_2028 = df_2028[feature_cols] # 2028 features (Trends from LSTM, Host info)
        
        # Grid Search
        print("  Running Grid Search CV...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1]
        }
        
        # Adjust Splits if samples are low (feature overlap issue)
        n_splits = 5
        if len(X_train) < 10:
            n_splits = 2
        
        xgb = XGBRegressor(objective='reg:squarederror', n_jobs=1)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=n_splits, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"  Best Params: {best_params}")
        
        # Bootstrap
        print("  Running Bootstrap (1000 iterations)...")
        n_iterations = 1000
        bootstrap_preds = np.zeros((len(X_pred_2028), n_iterations))
        
        X_train_reset = X_train.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        
        for i in range(n_iterations):
            if i % 500 == 0: print(f"    Iter {i}")
            # Resample
            indices = np.random.choice(len(X_train_reset), size=len(X_train_reset), replace=True)
            X_sample = X_train_reset.iloc[indices]
            y_sample = y_train_reset.iloc[indices]
            
            # Re-train
            model = XGBRegressor(**best_params, objective='reg:squarederror', n_jobs=1)
            model.fit(X_sample, y_sample)
            
            preds = model.predict(X_pred_2028)
            preds = np.maximum(preds, 0)
            bootstrap_preds[:, i] = preds
            
        # Stats
        means = np.mean(bootstrap_preds, axis=1)
        lowers = np.percentile(bootstrap_preds, 2.5, axis=1)
        uppers = np.percentile(bootstrap_preds, 97.5, axis=1)
        
        results_2028[f'Pred_{result_name}'] = np.round(means, 2)
        results_2028[f'Pred_{result_name}_Lower'] = np.round(lowers, 2)
        results_2028[f'Pred_{result_name}_Upper'] = np.round(uppers, 2)
        
    # Final Output
    final_output_path = os.path.join(processed_dir, '2028_medal_predictions.csv')
    results_2028.to_csv(final_output_path, index=False)
    print(f"\n2028 Predictions saved to {final_output_path}")
    
    if 'Pred_Gold' in results_2028.columns:
        results_2028 = results_2028.sort_values('Pred_Gold', ascending=False)
        print("\nTop 10 Predicted 2028 Gold Medals:")
        print(results_2028.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
