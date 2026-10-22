import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Load data
path = r"c:\Final Year ML Project\data\preprocessed_data().csv"
df = pd.read_csv(path)
# match notebook preprocessing
if 'Dropout_Risk' in df.columns:
    y = df['Dropout_Risk']
    X = df.drop(columns=['Dropout_Risk'])
else:
    raise ValueError('Dropout_Risk not found')

# one-hot encode clinician-coded categorical columns if present
for c in ['Faculty', 'Financial_Support_Source']:
    if c in X.columns:
        X = pd.get_dummies(X, columns=[c], drop_first=True)

bool_cols = X.select_dtypes(include='bool').columns
for c in bool_cols:
    X[c] = X[c].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=30,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1,
    gamma=0,
    eval_metric='logloss',
    random_state=42
)

cat = CatBoostClassifier(
    learning_rate=0.05,
    l2_leaf_reg=9,
    iterations=500,
    depth=10,
    border_count=64,
    verbose=False,
    random_state=42,
    auto_class_weights='Balanced'
)

models = {'Random Forest': rf, 'XGBoost': xgb, 'CatBoost': cat}
thresholds = np.arange(0.10, 0.91, 0.05)

rows = []
for name, base_model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        m = clone(base_model)
        m.fit(X_tr_res, y_tr_res)
        oof[va_idx] = m.predict_proba(X_va)[:, 1]

    curves = []
    for th in thresholds:
        pred = (oof >= th).astype(int)
        curves.append({
            'Threshold': round(float(th), 2),
            'Recall': recall_score(y_train, pred, zero_division=0),
            'Precision': precision_score(y_train, pred, zero_division=0),
            'F1': f1_score(y_train, pred, zero_division=0),
        })
    df_curve = pd.DataFrame(curves)
    selected = float(df_curve.loc[df_curve['F1'].idxmax(), 'Threshold'])
    print(f'\nModel: {name}')
    print(df_curve.sort_values('Threshold').to_string(index=False))
    print(f'Selected threshold by OOF F1: {selected}')

    # fit final model on full training data and evaluate once on the held-out test set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    model = clone(base_model)
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= selected).astype(int)
    row = {
        'Model': name,
        'SelectedThreshold': selected,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
    }
    rows.append(row)
    print(f'Test metrics: {row}')

print('\nFINAL COMPARISON')
print(pd.DataFrame(rows).sort_values(['Recall','F1'], ascending=False).to_string(index=False))
