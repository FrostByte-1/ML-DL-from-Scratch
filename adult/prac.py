import pandas as pd
import numpy as np

cols = [ "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]

train_df = pd.read_csv(
    "adult.data",
    names = cols,
    na_values = " ?",
    skipinitialspace = True
)

test_df = pd.read_csv(
    "adult.test",
    names = cols,
    na_values = " ?",
    skipinitialspace = True,
    skiprows = 1
)

test_df["income"] = test_df["income"].str.replace(".","",regex = False)

train_df = train_df.dropna()
test_df = test_df.dropna()

train_df["income"] = (train_df["income"] == ">50K").astype(int)
test_df["income"] = (test_df["income"] == ">50K").astype(int)

full_df = pd.concat([train_df, test_df])

cat_cols = full_df.select_dtypes(include="object").columns

full_df = pd.get_dummies(full_df, columns=cat_cols)

train_enc = full_df.iloc[:len(train_df)]
test_enc = full_df.iloc[len(train_df):]

X_train = train_enc.drop("income", axis = 1).values
y_train = train_enc["income"].values.reshape(-1,1)

X_test = test_enc.drop("income", axis = 1).values
y_test = test_enc["income"].values.reshape(-1,1)

def minmax_fit(X):
    mn = X.min(axis = 0)
    mx = X.max(axis = 0)
    return mn,mx

def minmax_transform(X,mn,mx):
    return (X - mn)/(mx - mn + 1e-8)

mn, mx = minmax_fit(X_train)

X_train_scaled = minmax_transform(X_train,mn,mx)
X_test_scaled = minmax_transform(X_test,mn,mx)

print("Train shape : ", X_train.shape)
print("Test shape : ", X_test.shape)

print("Scaled range : ",X_train_scaled.min(),X_train_scaled.max())