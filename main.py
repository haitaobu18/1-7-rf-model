import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import inspect

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("UTS.xlsx")

# =========================
# 2. 识别 1–7 系
# =========================
def get_series(x):
    try:
        return int(str(int(x))[0])
    except:
        return np.nan

df["Series"] = df["Alloy"].apply(get_series)
df = df[df["Series"].isin([1,2,3,4,5,6,7])]

# =========================
# 3. 特征与标签
# =========================
y = df["UTS"]
feature_cols = [c for c in df.columns if c not in ["UTS", "Series","Alloy"]]
X = df[feature_cols]
strata = df["Series"]

# =========================
# 4. 分层抽样（重点）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=strata,
    random_state=42
)

train_idx = X_train.index
test_idx = X_test.index

df_train = df.loc[train_idx]
df_test = df.loc[test_idx]

# =========================
# ★ 输出当前模型使用的特征（数值 + 类别）
# =========================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print("\n=== 使用的特征 ===")
print("数值特征:", numeric_features)
print("类别特征:", categorical_features)

# =========================
# ★ 输出每个系列抽取情况（训练/总量 + 百分比）
# =========================
print("\n=== 每个系列在训练集中占比 ===")
series_list = sorted(df["Series"].unique())

for s in series_list:
    total = len(df[df["Series"] == s])              # 总数
    train_n = len(df_train[df_train["Series"] == s])  # 训练集中数量
    pct = train_n / total * 100                     # 百分比

    print(f"{s} 系：{train_n} / {total} = {pct:.2f}%")

# =========================
# ★ 顺便也打印每个系列训练集/测试集有哪些样本
# =========================
print("\n=== 分层抽样结果展示（按系列） ===")
for s in series_list:
    train_subset = df_train[df_train["Series"] == s]
    test_subset = df_test[df_test["Series"] == s]

    print(f"\n--- {s} 系 ---")
    print(f"训练集 {len(train_subset)} 行，测试集 {len(test_subset)} 行")
    print("训练集 Alloy（去重）:", train_subset["Alloy"].unique())
    print("测试集 Alloy（去重）:", test_subset["Alloy"].unique())


# =========================
# 5. One-Hot + 标准化
# =========================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OHE, categorical_features)
    ]
)

# =========================
# 6. Random Forest
# =========================
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf)
])

model.fit(X_train, y_train)

print("\n模型训练完成！可以继续计算 R² / MSE 或画图。")
