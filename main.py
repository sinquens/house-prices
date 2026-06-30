import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# "yok" anlamına gelen NaN sütunları — kategoriye çevir
none_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
for col in none_cols:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")

# ordinal haritalar
quality_map      = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_exp_map     = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
bsmt_fin_map     = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
garage_fin_map   = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
lot_shape_map    = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
land_slope_map   = {"Sev": 1, "Mod": 2, "Gtl": 3}
paved_drive_map  = {"N": 0, "P": 1, "Y": 2}
functional_map   = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}

# None=0 anlamlı olan ordinal sütunlar (özellik yoksa sıfır mantıklı)
ordinal_with_none = {
    "ExterQual": quality_map, "ExterCond": quality_map,
    "BsmtQual": quality_map, "BsmtCond": quality_map,
    "HeatingQC": quality_map, "KitchenQual": quality_map,
    "FireplaceQu": quality_map, "GarageQual": quality_map, "GarageCond": quality_map,
    "PoolQC": quality_map,
    "BsmtExposure": bsmt_exp_map,
    "BsmtFinType1": bsmt_fin_map, "BsmtFinType2": bsmt_fin_map,
    "GarageFinish": garage_fin_map,
}

# her evde bulunan ordinal sütunlar — None=0 yok, NaN çıkarsa gerçek eksiktir
ordinal_no_none = {
    "LotShape": lot_shape_map,
    "LandSlope": land_slope_map,
    "PavedDrive": paved_drive_map,
}

for col, mapping in {**ordinal_with_none, **ordinal_no_none}.items():
    train[col] = train[col].map(mapping)
    test[col]  = test[col].map(mapping)

# Functional: data_description "aksi belirtilmedikçe Typ varsay" diyor
# train'de 0 eksik, test'te 2 var
train["Functional"] = train["Functional"].fillna("Typ").map(functional_map)
test["Functional"]  = test["Functional"].fillna("Typ").map(functional_map)

# sayısal görünümlü kategorik sütunlar
for col in ["MSSubClass", "MoSold", "YrSold"]:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

# GarageYrBlt: test'teki 2207 typo'sunu düzelt, sonra garaji olmayanları YearBuilt ile doldur
test["GarageYrBlt"] = test["GarageYrBlt"].replace(2207, 2007)
for df in [train, test]:
    mask = df["GarageYrBlt"].isnull()
    df.loc[mask, "GarageYrBlt"] = df.loc[mask, "YearBuilt"]

# --- durum raporu ---
import numpy as np

# --- Adım 1: Veri temizleme (devam) ---

# LotFrontage: Neighborhood medyanı ile doldur
lot_median = train.groupby("Neighborhood")["LotFrontage"].median()
for df in [train, test]:
    mask = df["LotFrontage"].isnull()
    df.loc[mask, "LotFrontage"] = df.loc[mask, "Neighborhood"].map(lot_median)

# MasVnrArea: eksik → 0 (MasVnrType zaten "None" yapıldı)
for df in [train, test]:
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# Sayısal bodrum/garaj sütunları: eksik → 0 (özellik yok demek)
zero_fill_num = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]
for col in zero_fill_num:
    for df in [train, test]:
        df[col] = df[col].fillna(0)

# Electrical: train'de 1 eksik → mode
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])

# MSZoning, SaleType, Exterior1st, Exterior2nd: test'te az eksik → mode
for col in ["MSZoning", "SaleType", "Exterior1st", "Exterior2nd", "KitchenQual", "Utilities"]:
    mode_val = train[col].mode()[0]
    test[col] = test[col].fillna(mode_val)

# Tanı: tüm eksikler giderildi mi?
remaining_train = train.isnull().sum()
remaining_test  = test.isnull().sum()
print("Kalan eksik (train):", remaining_train[remaining_train > 0].to_dict())
print("Kalan eksik (test):", remaining_test[remaining_test > 0].to_dict())

# --- Adım 2: Feature engineering ---

for df in [train, test]:
    df["TotalSF"]      = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"]     = df["YrSold"].astype(int) - df["YearBuilt"]
    df["RemodAge"]     = df["YrSold"].astype(int) - df["YearRemodAdd"]
    df["TotalBath"]    = (df["FullBath"] + 0.5 * df["HalfBath"]
                          + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorch"]   = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["HasPool"]      = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"]    = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"]      = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

# --- Adım 3: Target transform + birleştirme ---

y = np.log1p(train["SalePrice"])
train_ids = train["Id"]
test_ids  = test["Id"]

train.drop(columns=["SalePrice", "Id"], inplace=True)
test.drop(columns=["Id"], inplace=True)

n_train  = len(train)
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# --- Adım 4: One-hot encoding ---

all_data = pd.get_dummies(all_data)
train_enc = all_data[:n_train]
test_enc  = all_data[n_train:]

print(f"\nEğitim seti: {train_enc.shape}, Test seti: {test_enc.shape}")

# --- Adım 5: Model eğitimi + Cross-Validation ---

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ridge = Ridge(alpha=10)
scores = cross_val_score(ridge, train_enc, y, cv=5,
                         scoring="neg_root_mean_squared_error")
print(f"Ridge CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_scores = cross_val_score(xgb, train_enc, y, cv=5,
                                 scoring="neg_root_mean_squared_error")
    print(f"XGBoost CV RMSE: {-xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
    best_model = xgb if -xgb_scores.mean() < -scores.mean() else ridge
    best_name  = "XGBoost" if best_model is xgb else "Ridge"
except ImportError:
    print("xgboost kurulu değil, Ridge kullanılıyor.")
    best_model = ridge
    best_name  = "Ridge"

print(f"\nKullanılan model: {best_name}")

# --- Adım 6: Submission ---

best_model.fit(train_enc, y)
preds = np.expm1(best_model.predict(test_enc))

submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
submission.to_csv("submission.csv", index=False)
print(f"submission.csv kaydedildi ({len(submission)} satır)")
