import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Aykırı değer kaldır (GrLivArea > 4000 ve düşük fiyatlı 2 ev)
train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300_000))].reset_index(drop=True)

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
quality_map     = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_exp_map    = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
bsmt_fin_map    = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
garage_fin_map  = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
lot_shape_map   = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
land_slope_map  = {"Sev": 1, "Mod": 2, "Gtl": 3}
paved_drive_map = {"N": 0, "P": 1, "Y": 2}
functional_map  = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}

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
ordinal_no_none = {
    "LotShape": lot_shape_map,
    "LandSlope": land_slope_map,
    "PavedDrive": paved_drive_map,
}

for col, mapping in {**ordinal_with_none, **ordinal_no_none}.items():
    train[col] = train[col].map(mapping)
    test[col]  = test[col].map(mapping)

# Functional: "aksi belirtilmedikçe Typ varsay"
train["Functional"] = train["Functional"].fillna("Typ").map(functional_map)
test["Functional"]  = test["Functional"].fillna("Typ").map(functional_map)

# sayısal görünümlü kategorik sütunlar
for col in ["MSSubClass", "MoSold", "YrSold"]:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

# GarageYrBlt: test'teki 2207 typo'sunu düzelt, eksikleri YearBuilt ile doldur
test["GarageYrBlt"] = test["GarageYrBlt"].replace(2207, 2007)
for df in [train, test]:
    mask = df["GarageYrBlt"].isnull()
    df.loc[mask, "GarageYrBlt"] = df.loc[mask, "YearBuilt"]

# LotFrontage: Neighborhood medyanı ile doldur
lot_median = train.groupby("Neighborhood")["LotFrontage"].median()
for df in [train, test]:
    mask = df["LotFrontage"].isnull()
    df.loc[mask, "LotFrontage"] = df.loc[mask, "Neighborhood"].map(lot_median)

# MasVnrArea: eksik → 0
for df in [train, test]:
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# Sayısal bodrum/garaj eksikleri → 0
zero_fill_num = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]
for col in zero_fill_num:
    for df in [train, test]:
        df[col] = df[col].fillna(0)

# Kategorik eksikler → mode
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
for col in ["MSZoning", "SaleType", "Exterior1st", "Exterior2nd", "KitchenQual"]:
    test[col] = test[col].fillna(train[col].mode()[0])

# Utilities: train'de %99.9 AllPub → sabit sütun, modele bilgi katmaz
for df in [train, test]:
    df.drop(columns=["Utilities"], inplace=True)

# Genel catch-all: kalan sayısal eksikler → train medyanı, kategorik → train modu
num_cols = train.select_dtypes(include=[np.number]).columns.drop("SalePrice")
cat_cols = train.select_dtypes(include=["object"]).columns

for col in num_cols:
    if train[col].isnull().any() or test[col].isnull().any():
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

for col in cat_cols:
    if train[col].isnull().any() or test[col].isnull().any():
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col]  = test[col].fillna(mode_val)

# --- Feature engineering ---

for df in [train, test]:
    df["TotalSF"]      = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]   # toplam alan
    df["HouseAge"]     = df["YrSold"].astype(int) - df["YearBuilt"]            # ev yaşı (YrSold str→int)
    df["RemodAge"]     = df["YrSold"].astype(int) - df["YearRemodAdd"]
    df["TotalBath"]    = (df["FullBath"] + 0.5 * df["HalfBath"]                # toplam banyo
                          + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorch"]   = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["HasPool"]      = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"]    = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"]      = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

# --- Modelleme ---

y = np.log1p(train["SalePrice"])          # hedefi log dönüşümüne al
test_ids = test["Id"]

X      = train.drop(columns=["SalePrice", "Id"])
X_test = test.drop(columns=["Id"])

cat_cols = X.select_dtypes(include=["object"]).columns   # kalan kategorikler

# kategorikleri one-hot, sayısalları geçir
preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)

model = Pipeline([
    ("prep", preprocess),
    ("xgb", XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )),
])

# 5-fold CV RMSE (log ölçeği)
scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"XGBoost CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

# eğit → tahmin → geri dönüştür → kaydet
model.fit(X, y)
preds = np.expm1(model.predict(X_test))
submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
submission.to_csv("submission.csv", index=False)
print(f"submission.csv kaydedildi ({len(submission)} satır)")
