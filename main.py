import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import inspect


# =========================================================
# ★★ 评估指标函数（R²、MAE、RMSE）还有一个SHAP可解释性函数★★
# =========================================================
def evaluate_model(model, X_test, y_test, target_name):
    y_pred = model.predict(X_test)

    # 指标
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\n===== {target_name} 模型评估结果 =====")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    return r2, mae, rmse

# =========================================================
# ★★ 函数：根据不同 sheet 训练对应目标的模型 ★★
# =========================================================
def train_model_for_target(sheet_name, target_col):
    print("\n======================================")
    print(f"正在训练模型：{target_col}（读取 Sheet: {sheet_name}）")
    print("======================================")

    # 1. 读取对应 sheet
    df = pd.read_excel("YTS UTS EL sheet.xlsx", sheet_name=sheet_name)

    # 2. 识别 1–7 系
    def get_series(x):
        try:
            return int(str(int(x))[0])
        except:
            return np.nan

    df["Series"] = df["Alloy"].apply(get_series)
    df = df[df["Series"].isin([1,2,3,4,5,6,7])]

    # 3. 特征和标签
    y = df[target_col]
    feature_cols = [c for c in df.columns if c not in ["Series", "Alloy", target_col]]
    X = df[feature_cols]

    strata = df["Series"]

    # 4. 分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=strata,
        random_state=42
    )

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    # 5. 输出特征
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("\n=== 使用的特征 ===")
    print("数值特征:", numeric_features)
    print("类别特征:", categorical_features)

    # 6. 每个系列占比
    print("\n=== 每个系列在训练集中的占比 ===")
    for s in sorted(df["Series"].unique()):
        total = len(df[df["Series"] == s])
        train_n = len(df_train[df_train["Series"] == s])
        pct = train_n / total * 100
        print(f"{s} 系：{train_n} / {total} = {pct:.2f}%")

    # 7. 每个系列的 Alloy 分布
    print("\n=== 分层抽样结果（按系列） ===")
    for s in sorted(df["Series"].unique()):
        train_subset = df_train[df_train["Series"] == s]
        test_subset = df_test[df_test["Series"] == s]

        print(f"\n--- {s} 系 ---")
        print(f"训练集 {len(train_subset)} 行，测试集 {len(test_subset)} 行")
        print("训练集 Alloy（去重）:", train_subset["Alloy"].unique())
        print("测试集 Alloy（去重）:", test_subset["Alloy"].unique())

    # 8. 预处理
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

    # 9. 模型
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf)
    ])

    # 10. 训练
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, target_col)#模型评估

    print(f"\n模型训练完成：{target_col}（Sheet: {sheet_name}）")

    return model, X_test, y_test

# =========================================================
# ★★ 使用示例（你需要哪个模型就训练哪个）★★
# =========================================================

# 训练 UTS 模型
model_UTS, X_test_UTS, y_test_UTS = train_model_for_target("UTS", "UTS")

# 训练 YTS 模型
model_YTS, X_test_YTS, y_test_YTS = train_model_for_target("YTS", "YTS")

# 训练 EL 模型
model_EL, X_test_EL, y_test_EL = train_model_for_target("EL", "EL")
