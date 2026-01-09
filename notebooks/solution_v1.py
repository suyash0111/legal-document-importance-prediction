"""
Epstein Legal Documents - Importance Score Prediction
Kaggle Competition: bash-8-0-round-2

This notebook builds an ensemble model to predict document importance scores.
"""

import numpy as np
import pandas as pd
import warnings
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False
    print("CatBoost not installed, will use LightGBM + XGBoost only")

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


# =============================================================================
# Load Data
# =============================================================================

# Kaggle notebooks have data in /kaggle/input
if os.path.exists('/kaggle/input'):
    DATA_DIR = '/kaggle/input/bash-8-0-round-2'
else:
    DATA_DIR = './'

print(f"Loading data from: {DATA_DIR}")

# Check what files we have
files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
print(f"Available files: {files}")

# Find the right file names
def find_file(keywords, file_list):
    for f in file_list:
        if all(k in f.lower() for k in keywords):
            return f
    return None

train_file = find_file(['train'], files) or 'train.csv'
test_file = find_file(['test'], files) or 'test.csv'

train = pd.read_csv(os.path.join(DATA_DIR, train_file))
test = pd.read_csv(os.path.join(DATA_DIR, test_file))

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Store IDs for submission
test_ids = test['id'].copy()
target = train['Importance Score'].values

print(f"Target mean: {target.mean():.2f}, std: {target.std():.2f}")


# =============================================================================
# Preprocessing
# =============================================================================

TEXT_COLS = ['Headline', 'Key Insights', 'Reasoning']
LIST_COLS = ['Power Mentions', 'Agencies', 'Tags']

def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess(df):
    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    if 'Lead Types' in df.columns:
        df['Lead Types'] = df['Lead Types'].fillna('unknown').astype(str)
    if 'Source File' in df.columns:
        df['Source File'] = df['Source File'].fillna('unknown').astype(str)
    return df

train = preprocess(train)
test = preprocess(test)


# =============================================================================
# Feature Engineering
# =============================================================================

def count_items(text):
    """Count semicolon-separated items"""
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return len([x.strip() for x in str(text).split(';') if x.strip()])

# Keywords that might indicate important documents
KEYWORDS = [
    'epstein', 'maxwell', 'abuse', 'trafficking', 'minor', 'victim',
    'island', 'flight', 'investigation', 'fbi', 'doj', 'settlement',
    'lawsuit', 'allegation', 'witness', 'testimony', 'evidence',
    'confidential', 'sealed', 'redacted', 'prosecution', 'indictment'
]

def create_features(df):
    df = df.copy()
    
    # Concatenate text fields
    df['full_text'] = (
        df['Headline'].fillna('') + ' ' +
        df['Key Insights'].fillna('') + ' ' +
        df['Reasoning'].fillna('')
    ).str.strip()
    
    df['full_text_clean'] = df['full_text'].apply(clean_text)
    
    # Count features
    df['n_power_mentions'] = df['Power Mentions'].apply(count_items)
    df['n_agencies'] = df['Agencies'].apply(count_items)
    df['n_tags'] = df['Tags'].apply(count_items) if 'Tags' in df.columns else 0
    
    # Text length features
    for col in TEXT_COLS:
        if col in df.columns:
            name = col.lower().replace(' ', '_')
            df[f'{name}_len'] = df[col].str.len()
            df[f'{name}_words'] = df[col].str.split().str.len().fillna(0)
    
    # Full text stats
    df['text_len'] = df['full_text'].str.len()
    df['text_words'] = df['full_text'].str.split().str.len().fillna(0)
    df['text_sentences'] = df['full_text'].str.count(r'[.!?]')
    
    words_list = df['full_text'].str.split()
    df['avg_word_len'] = words_list.apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
    df['unique_words'] = words_list.apply(lambda x: len(set(x)) if x else 0)
    
    # Keyword counts
    text_lower = df['full_text'].str.lower()
    df['keyword_count'] = text_lower.apply(
        lambda x: sum(1 for kw in KEYWORDS if kw in str(x))
    )
    
    # Binary flags for important terms
    df['has_epstein'] = text_lower.str.contains('epstein', na=False).astype(int)
    df['has_maxwell'] = text_lower.str.contains('maxwell', na=False).astype(int)
    df['has_victim'] = text_lower.str.contains('victim|minor|abuse', regex=True, na=False).astype(int)
    df['has_legal'] = text_lower.str.contains('lawsuit|settlement|court', regex=True, na=False).astype(int)
    
    # Agency flags
    agencies = df['Agencies'].str.lower()
    df['has_fbi'] = agencies.str.contains('fbi', na=False).astype(int)
    df['has_doj'] = agencies.str.contains('doj|justice', regex=True, na=False).astype(int)
    
    # Lead type flags
    leads = df['Lead Types'].str.lower()
    df['is_sexual'] = leads.str.contains('sexual', na=False).astype(int)
    df['is_financial'] = leads.str.contains('financial', na=False).astype(int)
    df['is_obstruction'] = leads.str.contains('obstruction', na=False).astype(int)
    df['n_lead_types'] = df['Lead Types'].apply(count_items)
    
    # Ratios
    df['mentions_ratio'] = df['n_power_mentions'] / (df['text_words'] + 1)
    df['keyword_ratio'] = df['keyword_count'] / (df['text_words'] + 1)
    df['unique_ratio'] = df['unique_words'] / (df['text_words'] + 1)
    
    return df

train = create_features(train)
test = create_features(test)

print(f"Features created")


# =============================================================================
# TF-IDF + SVD
# =============================================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    stop_words='english',
    sublinear_tf=True,
    dtype=np.float32
)

train_tfidf = tfidf.fit_transform(train['full_text_clean'])
test_tfidf = tfidf.transform(test['full_text_clean'])

print(f"TF-IDF shape: {train_tfidf.shape}")

# Reduce dimensions with SVD
N_COMPONENTS = 100
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED, n_iter=15)

train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")

svd_cols = [f'svd_{i}' for i in range(N_COMPONENTS)]
train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train.index)
test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test.index)


# =============================================================================
# Target Encoding
# =============================================================================

def target_encode(train_df, test_df, col, target_col, n_splits=5, smooth=20):
    """Target encoding with CV to avoid leakage"""
    encoded_train = np.zeros(len(train_df))
    global_mean = train_df[target_col].mean()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for tr_idx, val_idx in kf.split(train_df):
        fold_df = train_df.iloc[tr_idx]
        stats = fold_df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
        encoded_train[val_idx] = train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
    
    # Full data for test
    stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
    smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
    encoded_test = test_df[col].map(smoothed).fillna(global_mean).values
    
    return encoded_train, encoded_test

train['lead_type_enc'], test['lead_type_enc'] = target_encode(
    train, test, 'Lead Types', 'Importance Score'
)

# Label encoding as extra feature
le = LabelEncoder()
all_leads = pd.concat([train['Lead Types'], test['Lead Types']])
le.fit(all_leads)
train['lead_type_le'] = le.transform(train['Lead Types'])
test['lead_type_le'] = le.transform(test['Lead Types'])


# =============================================================================
# Build Feature Matrix
# =============================================================================

FEATURES = [
    'n_power_mentions', 'n_agencies', 'n_tags', 'n_lead_types',
    'headline_len', 'headline_words',
    'key_insights_len', 'key_insights_words',
    'reasoning_len', 'reasoning_words',
    'text_len', 'text_words', 'text_sentences', 'avg_word_len', 'unique_words',
    'keyword_count', 'has_epstein', 'has_maxwell', 'has_victim', 'has_legal',
    'has_fbi', 'has_doj', 'is_sexual', 'is_financial', 'is_obstruction',
    'mentions_ratio', 'keyword_ratio', 'unique_ratio',
    'lead_type_enc', 'lead_type_le'
]

# Keep only columns that exist
FEATURES = [f for f in FEATURES if f in train.columns]

train_num = train[FEATURES].fillna(0).astype(np.float32)
test_num = test[FEATURES].fillna(0).astype(np.float32)

X_train = pd.concat([train_num.reset_index(drop=True), train_svd_df.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_num.reset_index(drop=True), test_svd_df.reset_index(drop=True)], axis=1)
y_train = target

print(f"Final feature matrix: {X_train.shape}")


# =============================================================================
# Model Training
# =============================================================================

# Predictions storage
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train)) if USE_CATBOOST else None

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test)) if USE_CATBOOST else None

scores_lgb = []
scores_xgb = []
scores_cat = [] if USE_CATBOOST else None

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1
}

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"\nTraining {N_FOLDS}-fold models...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model_lgb = lgb.train(
        lgb_params, lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    pred_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / N_FOLDS
    rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    scores_lgb.append(rmse_lgb)
    print(f"  LightGBM RMSE: {rmse_lgb:.5f}")
    
    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model_xgb = xgb.train(
        xgb_params, dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=150,
        verbose_eval=500
    )
    
    oof_xgb[val_idx] = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration + 1))
    pred_xgb += model_xgb.predict(xgb.DMatrix(X_test), iteration_range=(0, model_xgb.best_iteration + 1)) / N_FOLDS
    rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    scores_xgb.append(rmse_xgb)
    print(f"  XGBoost RMSE: {rmse_xgb:.5f}")
    
    # CatBoost
    if USE_CATBOOST:
        model_cat = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3,
            random_seed=SEED,
            early_stopping_rounds=150,
            verbose=500
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        oof_cat[val_idx] = model_cat.predict(X_val)
        pred_cat += model_cat.predict(X_test) / N_FOLDS
        rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
        scores_cat.append(rmse_cat)
        print(f"  CatBoost RMSE: {rmse_cat:.5f}")


# =============================================================================
# Ensemble
# =============================================================================

print("\n" + "=" * 50)
print("CV Results:")
print(f"  LightGBM: {np.mean(scores_lgb):.5f} +/- {np.std(scores_lgb):.5f}")
print(f"  XGBoost:  {np.mean(scores_xgb):.5f} +/- {np.std(scores_xgb):.5f}")

if USE_CATBOOST:
    print(f"  CatBoost: {np.mean(scores_cat):.5f} +/- {np.std(scores_cat):.5f}")
    
    # Weight by inverse RMSE
    w1 = 1 / np.mean(scores_lgb)
    w2 = 1 / np.mean(scores_xgb)
    w3 = 1 / np.mean(scores_cat)
    total = w1 + w2 + w3
    w1, w2, w3 = w1/total, w2/total, w3/total
    
    print(f"\nEnsemble weights: LGB={w1:.3f}, XGB={w2:.3f}, CAT={w3:.3f}")
    
    oof_final = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    pred_final = w1 * pred_lgb + w2 * pred_xgb + w3 * pred_cat
else:
    w1 = 1 / np.mean(scores_lgb)
    w2 = 1 / np.mean(scores_xgb)
    total = w1 + w2
    w1, w2 = w1/total, w2/total
    
    print(f"\nEnsemble weights: LGB={w1:.3f}, XGB={w2:.3f}")
    
    oof_final = w1 * oof_lgb + w2 * oof_xgb
    pred_final = w1 * pred_lgb + w2 * pred_xgb

final_rmse = np.sqrt(mean_squared_error(y_train, oof_final))
print(f"\nFinal ensemble CV RMSE: {final_rmse:.5f}")


# =============================================================================
# Generate Submission
# =============================================================================

# Clip to valid range
predictions = np.clip(pred_final, 0, 100)

submission = pd.DataFrame({
    'id': test_ids,
    'Importance Score': predictions
})

# Save
output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else './'
output_path = os.path.join(output_dir, 'submission.csv')
submission.to_csv(output_path, index=False)

print(f"\nSubmission saved to: {output_path}")
print(f"Shape: {submission.shape}")
print(f"Predictions - min: {predictions.min():.2f}, max: {predictions.max():.2f}, mean: {predictions.mean():.2f}")
