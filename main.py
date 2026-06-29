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
print("Kalan eksik (train):")
mt = train.isnull().sum()
print(mt[mt > 0])

print("\nKalan eksik (test):")
ms = test.isnull().sum()
print(ms[ms > 0])

# MasVnrType ve MasVnrArea aynı satırlarda mı eksik?
area_null = train["MasVnrArea"].isnull()
print(f"\nMasVnrArea eksik: {area_null.sum()} satır")
print(f"Bu satırlarda MasVnrType: {train.loc[area_null, 'MasVnrType'].unique()}")
