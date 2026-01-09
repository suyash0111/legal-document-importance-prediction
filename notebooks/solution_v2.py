"""
Epstein Legal Documents - Improved Solution v2
Kaggle Competition: bash-8-0-round-2

Improvements over v1:
- Increased SVD components (150)
- Neural network embeddings (if available)
- Ridge stacking meta-learner
- Tuned hyperparameters
- Additional text features
"""

import numpy as np
import pandas as pd
import warnings
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False

# Try sentence transformers for better embeddings
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except ImportError:
    USE_SBERT = False
    print("sentence-transformers not available, using TF-IDF only")

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


# =============================================================================
# Load Data
# =============================================================================

if os.path.exists('/kaggle/input'):
    DATA_DIR = '/kaggle/input/bash-8-0-round-2'
else:
    DATA_DIR = './'

print(f"Loading data from: {DATA_DIR}")

files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []

def find_file(keywords, file_list):
    for f in file_list:
        if all(k in f.lower() for k in keywords):
            return f
    return None

train_file = find_file(['train'], files) or 'train.csv'
test_file = find_file(['test'], files) or 'test.csv'

train = pd.read_csv(os.path.join(DATA_DIR, train_file))
test = pd.read_csv(os.path.join(DATA_DIR, test_file))

print(f"Train: {train.shape}, Test: {test.shape}")

test_ids = test['id'].copy()
target = train['Importance Score'].values


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
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return len([x.strip() for x in str(text).split(';') if x.strip()])

KEYWORDS = [
    'epstein', 'maxwell', 'abuse', 'trafficking', 'minor', 'victim',
    'island', 'flight', 'investigation', 'fbi', 'doj', 'settlement',
    'lawsuit', 'allegation', 'witness', 'testimony', 'evidence',
    'confidential', 'sealed', 'redacted', 'prosecution', 'indictment',
    'prince', 'andrew', 'clinton', 'trump', 'acosta', 'dershowitz',
    'wexner', 'money', 'payment', 'bank', 'wire', 'transfer'
]

def create_features(df):
    df = df.copy()
    
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
    
    # Text length
    for col in TEXT_COLS:
        if col in df.columns:
            name = col.lower().replace(' ', '_')
            df[f'{name}_len'] = df[col].str.len()
            df[f'{name}_words'] = df[col].str.split().str.len().fillna(0)
    
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
    
    # Binary flags
    df['has_epstein'] = text_lower.str.contains('epstein', na=False).astype(int)
    df['has_maxwell'] = text_lower.str.contains('maxwell', na=False).astype(int)
    df['has_victim'] = text_lower.str.contains('victim|minor|abuse', regex=True, na=False).astype(int)
    df['has_legal'] = text_lower.str.contains('lawsuit|settlement|court', regex=True, na=False).astype(int)
    df['has_financial'] = text_lower.str.contains('money|bank|payment|wire', regex=True, na=False).astype(int)
    df['has_celebrity'] = text_lower.str.contains('prince|clinton|trump', regex=True, na=False).astype(int)
    
    agencies = df['Agencies'].str.lower()
    df['has_fbi'] = agencies.str.contains('fbi', na=False).astype(int)
    df['has_doj'] = agencies.str.contains('doj|justice', regex=True, na=False).astype(int)
    
    leads = df['Lead Types'].str.lower()
    df['is_sexual'] = leads.str.contains('sexual', na=False).astype(int)
    df['is_financial'] = leads.str.contains('financial', na=False).astype(int)
    df['is_obstruction'] = leads.str.contains('obstruction', na=False).astype(int)
    df['n_lead_types'] = df['Lead Types'].apply(count_items)
    
    # Ratios
    df['mentions_ratio'] = df['n_power_mentions'] / (df['text_words'] + 1)
    df['keyword_ratio'] = df['keyword_count'] / (df['text_words'] + 1)
    df['unique_ratio'] = df['unique_words'] / (df['text_words'] + 1)
    
    # Log transforms for skewed features
    df['log_text_len'] = np.log1p(df['text_len'])
    df['log_mentions'] = np.log1p(df['n_power_mentions'])
    
    return df

train = create_features(train)
test = create_features(test)
print("Features created")


# =============================================================================
# Sentence Transformer Embeddings (if available)
# =============================================================================

if USE_SBERT:
    print("Generating sentence embeddings...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    train_embeddings = sbert_model.encode(
        train['full_text_clean'].tolist(), 
        show_progress_bar=True,
        batch_size=64
    )
    test_embeddings = sbert_model.encode(
        test['full_text_clean'].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    
    # Reduce dimensions with PCA to avoid overfitting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=SEED)
    train_sbert = pca.fit_transform(train_embeddings)
    test_sbert = pca.transform(test_embeddings)
    
    sbert_cols = [f'sbert_{i}' for i in range(50)]
    train_sbert_df = pd.DataFrame(train_sbert, columns=sbert_cols, index=train.index)
    test_sbert_df = pd.DataFrame(test_sbert, columns=sbert_cols, index=test.index)
    print(f"SBERT embeddings: {train_sbert.shape}")
else:
    train_sbert_df = pd.DataFrame()
    test_sbert_df = pd.DataFrame()


# =============================================================================
# TF-IDF + SVD (increased components)
# =============================================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
    stop_words='english',
    sublinear_tf=True,
    dtype=np.float32
)

train_tfidf = tfidf.fit_transform(train['full_text_clean'])
test_tfidf = tfidf.transform(test['full_text_clean'])

N_COMPONENTS = 150
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED, n_iter=20)

train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

print(f"TF-IDF SVD: {train_svd.shape}, variance: {svd.explained_variance_ratio_.sum():.4f}")

svd_cols = [f'svd_{i}' for i in range(N_COMPONENTS)]
train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train.index)
test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test.index)


# =============================================================================
# Target Encoding
# =============================================================================

def target_encode(train_df, test_df, col, target_col, n_splits=5, smooth=20):
    encoded_train = np.zeros(len(train_df))
    global_mean = train_df[target_col].mean()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for tr_idx, val_idx in kf.split(train_df):
        fold_df = train_df.iloc[tr_idx]
        stats = fold_df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
        encoded_train[val_idx] = train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
    
    stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
    smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
    encoded_test = test_df[col].map(smoothed).fillna(global_mean).values
    
    return encoded_train, encoded_test

train['lead_type_enc'], test['lead_type_enc'] = target_encode(
    train, test, 'Lead Types', 'Importance Score'
)

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
    'keyword_count', 
    'has_epstein', 'has_maxwell', 'has_victim', 'has_legal', 'has_financial', 'has_celebrity',
    'has_fbi', 'has_doj', 
    'is_sexual', 'is_financial', 'is_obstruction',
    'mentions_ratio', 'keyword_ratio', 'unique_ratio',
    'log_text_len', 'log_mentions',
    'lead_type_enc', 'lead_type_le'
]

FEATURES = [f for f in FEATURES if f in train.columns]

train_num = train[FEATURES].fillna(0).astype(np.float32)
test_num = test[FEATURES].fillna(0).astype(np.float32)

# Combine all features
dfs_train = [train_num.reset_index(drop=True), train_svd_df.reset_index(drop=True)]
dfs_test = [test_num.reset_index(drop=True), test_svd_df.reset_index(drop=True)]

if USE_SBERT:
    dfs_train.append(train_sbert_df.reset_index(drop=True))
    dfs_test.append(test_sbert_df.reset_index(drop=True))

X_train = pd.concat(dfs_train, axis=1)
X_test = pd.concat(dfs_test, axis=1)
y_train = target

print(f"Feature matrix: {X_train.shape}")


# =============================================================================
# Model Training with Optimized Parameters
# =============================================================================

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train)) if USE_CATBOOST else None

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test)) if USE_CATBOOST else None

scores_lgb = []
scores_xgb = []
scores_cat = [] if USE_CATBOOST else None

# Tuned LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 48,
    'max_depth': 7,
    'min_child_samples': 25,
    'subsample': 0.75,
    'subsample_freq': 1,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1
}

# Tuned XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
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
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    pred_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / N_FOLDS
    rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    scores_lgb.append(rmse_lgb)
    print(f"  LightGBM: {rmse_lgb:.5f}")
    
    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model_xgb = xgb.train(
        xgb_params, dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=200,
        verbose_eval=500
    )
    
    oof_xgb[val_idx] = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration + 1))
    pred_xgb += model_xgb.predict(xgb.DMatrix(X_test), iteration_range=(0, model_xgb.best_iteration + 1)) / N_FOLDS
    rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    scores_xgb.append(rmse_xgb)
    print(f"  XGBoost: {rmse_xgb:.5f}")
    
    # CatBoost
    if USE_CATBOOST:
        model_cat = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=5,
            random_seed=SEED,
            early_stopping_rounds=200,
            verbose=500
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        oof_cat[val_idx] = model_cat.predict(X_val)
        pred_cat += model_cat.predict(X_test) / N_FOLDS
        rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
        scores_cat.append(rmse_cat)
        print(f"  CatBoost: {rmse_cat:.5f}")


# =============================================================================
# Stacking with Ridge Meta-Learner
# =============================================================================

print("\n" + "=" * 50)
print("CV Results:")
print(f"  LightGBM: {np.mean(scores_lgb):.5f} +/- {np.std(scores_lgb):.5f}")
print(f"  XGBoost:  {np.mean(scores_xgb):.5f} +/- {np.std(scores_xgb):.5f}")

if USE_CATBOOST:
    print(f"  CatBoost: {np.mean(scores_cat):.5f} +/- {np.std(scores_cat):.5f}")
    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    test_stack = np.column_stack([pred_lgb, pred_xgb, pred_cat])
else:
    oof_stack = np.column_stack([oof_lgb, oof_xgb])
    test_stack = np.column_stack([pred_lgb, pred_xgb])

# Train Ridge meta-learner with CV
print("\nTraining Ridge stacking meta-learner...")
meta_oof = np.zeros(len(X_train))
meta_pred = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack)):
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_stack[tr_idx], y_train[tr_idx])
    meta_oof[val_idx] = meta_model.predict(oof_stack[val_idx])

# Final meta model on all data
final_meta = Ridge(alpha=1.0)
final_meta.fit(oof_stack, y_train)
meta_pred = final_meta.predict(test_stack)

stacked_rmse = np.sqrt(mean_squared_error(y_train, meta_oof))
print(f"Stacked CV RMSE: {stacked_rmse:.5f}")

# Also compute simple weighted ensemble for comparison
if USE_CATBOOST:
    w1 = 1 / np.mean(scores_lgb)
    w2 = 1 / np.mean(scores_xgb)
    w3 = 1 / np.mean(scores_cat)
    total = w1 + w2 + w3
    w1, w2, w3 = w1/total, w2/total, w3/total
    oof_weighted = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    pred_weighted = w1 * pred_lgb + w2 * pred_xgb + w3 * pred_cat
else:
    w1 = 1 / np.mean(scores_lgb)
    w2 = 1 / np.mean(scores_xgb)
    total = w1 + w2
    w1, w2 = w1/total, w2/total
    oof_weighted = w1 * oof_lgb + w2 * oof_xgb
    pred_weighted = w1 * pred_lgb + w2 * pred_xgb

weighted_rmse = np.sqrt(mean_squared_error(y_train, oof_weighted))
print(f"Weighted CV RMSE: {weighted_rmse:.5f}")

# Use better approach
if stacked_rmse < weighted_rmse:
    print("Using stacked predictions")
    final_pred = meta_pred
    final_rmse = stacked_rmse
else:
    print("Using weighted predictions")
    final_pred = pred_weighted
    final_rmse = weighted_rmse

print(f"\nFinal CV RMSE: {final_rmse:.5f}")


# =============================================================================
# Generate Submission
# =============================================================================

predictions = np.clip(final_pred, 0, 100)

submission = pd.DataFrame({
    'id': test_ids,
    'Importance Score': predictions
})

output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else './'
output_path = os.path.join(output_dir, 'submission.csv')
submission.to_csv(output_path, index=False)

print(f"\nSubmission saved to: {output_path}")
print(f"Shape: {submission.shape}")
print(f"Predictions - min: {predictions.min():.2f}, max: {predictions.max():.2f}, mean: {predictions.mean():.2f}")
