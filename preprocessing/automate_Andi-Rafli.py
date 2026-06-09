"""
automate_Andi_Rafli.py
======================
Skrip otomatisasi preprocessing dataset Heart Disease UCI.
Mengkonversi workflow eksperimen dari Eksperimen_Andi_Rafli.ipynb
menjadi fungsi yang dapat dijalankan secara otomatis.

Penggunaan:
    python automate_Andi_Rafli.py
    python automate_Andi_Rafli.py --input heart_disease_uci.csv --output_dir heart_disease_preprocessing
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✅ Dataset dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")
    return df


# ──────────────────────────────────────────────────────────────
# 2. STANDARISASI KOLOM
# ──────────────────────────────────────────────────────────────
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi nama kolom dan tipe data agar konsisten
    dengan format yang digunakan pada tahap eksperimen.
    """
    df = df.copy()

    # Hapus kolom tidak diperlukan
    drop_cols = ['id', 'dataset']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename kolom
    rename_map = {
        'thalch': 'thalach',
        'num': 'condition',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Binarisasi target
    if 'condition' in df.columns:
        df['condition'] = (df['condition'] > 0).astype(int)

    # Encode sex
    if df['sex'].dtype == object:
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    # Encode cp
    cp_map = {
        'typical angina': 0, 'atypical angina': 1,
        'non-anginal': 2, 'asymptomatic': 3
    }
    if df['cp'].dtype == object:
        df['cp'] = df['cp'].map(cp_map)

    # Encode fbs & exang
    for col in ['fbs', 'exang']:
        if df[col].dtype == object or df[col].dtype == bool:
            df[col] = df[col].map(
                {True: 1, False: 0, 'True': 1, 'False': 0}
            ).astype(float)

    # Encode restecg
    restecg_map = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
    if df['restecg'].dtype == object:
        df['restecg'] = df['restecg'].str.lower().map(restecg_map)

    # Encode slope
    slope_map = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
    if df['slope'].dtype == object:
        df['slope'] = df['slope'].str.lower().map(slope_map)

    # Encode thal
    thal_map = {'normal': 1, 'fixed defect': 2, 'reversable defect': 3}
    if df['thal'].dtype == object:
        df['thal'] = df['thal'].str.lower().map(thal_map)

    print("✅ Standarisasi kolom selesai")
    return df


# ──────────────────────────────────────────────────────────────
# 3. HANDLING MISSING VALUES
# ──────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputasi missing values menggunakan median (robust terhadap outlier)."""
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != 'condition']

    missing_before = df[feat_cols].isnull().sum().sum()

    imputer = SimpleImputer(strategy='median')
    df[feat_cols] = imputer.fit_transform(df[feat_cols])

    missing_after = df[feat_cols].isnull().sum().sum()
    print(f"✅ Missing values: {missing_before} → {missing_after}")
    return df


# ──────────────────────────────────────────────────────────────
# 4. HANDLING DUPLICATES
# ──────────────────────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus data duplikat."""
    dup_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    dup_after = df.duplicated().sum()
    print(f"✅ Duplikat: {dup_before} → {dup_after} | Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# 5. HANDLING OUTLIERS (WINSORIZATION)
# ──────────────────────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani outlier dengan metode Winsorization (IQR capping)."""
    df = df.copy()
    outlier_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']

    for col in outlier_cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"   {col:<12}: {n_out} outlier di-cap ke [{lower:.1f}, {upper:.1f}]")

    print("✅ Penanganan outlier selesai")
    return df


# ──────────────────────────────────────────────────────────────
# 6. SPLIT & SCALE
# ──────────────────────────────────────────────────────────────
def split_and_scale(df: pd.DataFrame):
    """
    Memisahkan fitur dan target, train-test split 80:20,
    lalu standarisasi fitur numerik.

    Returns:
        X_train, X_test, y_train, y_test (semua sebagai DataFrame/Series)
    """
    X = df.drop('condition', axis=1)
    y = df['condition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scale_cols = [c for c in scale_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    print(f"✅ Split selesai — Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# 7. SAVE OUTPUT
# ──────────────────────────────────────────────────────────────
def save_output(X_train, X_test, y_train, y_test, output_dir: str):
    """Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['condition'] = y_train.values
    test_df  = X_test.copy()
    test_df['condition']  = y_test.values

    train_path = os.path.join(output_dir, 'heart_disease_preprocessing_train.csv')
    test_path  = os.path.join(output_dir, 'heart_disease_preprocessing_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"✅ File disimpan:")
    print(f"   → {train_path} ({train_df.shape[0]} baris)")
    print(f"   → {test_path}  ({test_df.shape[0]} baris)")


# ──────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def preprocess(input_file: str, output_dir: str):
    """Pipeline preprocessing lengkap dari raw → siap latih."""
    print("=" * 60)
    print("  PIPELINE PREPROCESSING — Heart Disease UCI")
    print("=" * 60)

    df = load_data(input_file)
    df = standardize_columns(df)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = handle_outliers(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)
    save_output(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 60)
    print("  ✅ PREPROCESSING SELESAI")
    print("=" * 60)
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate preprocessing Heart Disease UCI')
    parser.add_argument('--input',      type=str, default='heart_disease_uci.csv',
                        help='Path ke file CSV raw dataset')
    parser.add_argument('--output_dir', type=str, default='heart_disease_preprocessing',
                        help='Folder output hasil preprocessing')
    args = parser.parse_args()

    preprocess(args.input, args.output_dir)
