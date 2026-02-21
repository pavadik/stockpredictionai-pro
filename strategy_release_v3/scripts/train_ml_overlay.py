import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def train_xgboost_overlay():
    dataset_path = os.path.join(OUT_DIR, "block_e_ml_dataset.csv")
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["entry_time"])
    
    # Split train/test (chronological)
    # We use 2007-2018 for training, 2019+ for testing to match our strategy OOS
    train_end_dt = pd.Timestamp("2018-12-31")
    
    df_train = df[df["entry_time"] <= train_end_dt].copy()
    df_test = df[df["entry_time"] > train_end_dt].copy()
    
    print(f"Train samples (<=2018): {len(df_train)}")
    print(f"Test samples (>=2019): {len(df_test)}")
    
    # Features to use for training
    features = [
        "hour", "minute", "day_of_week", "month",
        "d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d",
        "m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m",
        "is_long"
    ]
    target = "target"
    
    X_train = df_train[features]
    y_train = df_train[target]
    
    X_test = df_test[features]
    y_test = df_test[target]
    
    print(f"\nTrain Target Distribution: {y_train.mean():.2%} positive")
    print(f"Test Target Distribution: {y_test.mean():.2%} positive")
    
    # Calculate scale_pos_weight to handle class imbalance
    # We want to predict positive trades, but they are minority (~32%)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    print("\nTraining XGBoost model...")
    # Initialize XGBoost Classifier
    # Tuning parameters slightly for a noisy financial dataset
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=4,         # Shallow trees to prevent overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        early_stopping_rounds=20
    )
    
    # Create evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Fit the model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    print("\nEvaluating model on OOS Test Set...")
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Let's see the distribution of predictions
    print(f"Predicted probabilities: Mean={y_pred_proba.mean():.3f}, Min={y_pred_proba.min():.3f}, Max={y_pred_proba.max():.3f}")
    
    # Calculate metrics at different thresholds
    # Strategy: if model output < threshold -> don't take the trade (filter it)
    # We want high precision for the trades we DO take.
    
    thresholds = [0.4, 0.5, 0.55, 0.6, 0.65]
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        trades_taken = y_pred.sum()
        trades_skipped = len(y_pred) - trades_taken
        pct_taken = trades_taken / len(y_pred)
        
        if trades_taken > 0:
            precision = precision_score(y_test, y_pred)
            
            # Simulated impact
            # We look at the actual PnL of trades we would have taken vs skipped
            df_test_eval = df_test.copy()
            df_test_eval["ml_pred"] = y_pred
            
            taken_pnl = df_test_eval[df_test_eval["ml_pred"] == 1]["pnl_net"].sum()
            skipped_pnl = df_test_eval[df_test_eval["ml_pred"] == 0]["pnl_net"].sum()
            total_orig_pnl = df_test_eval["pnl_net"].sum()
            
            print(f"\n--- Threshold {thresh} ---")
            print(f"Trades taken: {trades_taken} ({pct_taken:.1%}) | Skipped: {trades_skipped}")
            print(f"Precision (Win Rate of taken trades): {precision:.1%}")
            print(f"Orig PnL: {total_orig_pnl:,.0f} -> ML PnL: {taken_pnl:,.0f} (Improvement: {taken_pnl - total_orig_pnl:,.0f})")
            print(f"PnL of skipped trades: {skipped_pnl:,.0f}")
        else:
            print(f"\n--- Threshold {thresh} ---")
            print("No trades taken.")
            
    # Feature Importance
    print("\nFeature Importances (Gain):")
    importance = model.get_booster().get_score(importance_type="gain")
    # Sort by value
    importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    for feat, gain in importance.items():
        print(f"  {feat}: {gain:.2f}")
        
    # Save the model
    model_path = os.path.join(OUT_DIR, "block_e_xgboost_overlay.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save test dataset with predictions for further analysis
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    
    df_train["ml_proba"] = y_train_pred_proba
    df_test["ml_proba"] = y_pred_proba
    
    df_all = pd.concat([df_train, df_test])
    df_all.to_csv(os.path.join(OUT_DIR, "block_e_ml_test_results.csv"), index=False)
    print(f"Saved test results to {os.path.join(OUT_DIR, 'block_e_ml_test_results.csv')}")

if __name__ == "__main__":
    train_xgboost_overlay()