import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import inspect


# =========================================================
# ★★ 评估函数（R²、MAE、RMSE）★★
# =========================================================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"\n===== {name} 模型评估结果 =====")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    return r2, mae, rmse



# =========================================================
# ★★ 自动训练全部 21 个模型（从单个 Excel 文件读取）★★
# =========================================================
def train_all_series_models_from_sheet():

    configs = [
        ("UTS", "UTS"),
        ("YTS", "YTS"),
        ("EL",  "EL")
    ]

    excel_file = "YTS UTS EL sheet.xlsx"

    all_models = {}
    all_scores = { "UTS":{}, "YTS":{}, "EL":{} }

    for sheet_name, target_col in configs:

        print("\n========================================")
        print(f"开始训练 {target_col} 的 7 个系列模型 (sheet: {sheet_name})")
        print("========================================")

        # 读取 sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # 识别系列
        def get_series(x):
            try:
                return int(str(int(x))[0])
            except:
                return np.nan

        df["Series"] = df["Alloy"].apply(get_series)
        df = df[df["Series"].isin([1,2,3,4,5,6,7])]

        # 针对该目标训练 7 个模型
        for s in range(1, 8):

            print(f"\n-------------------------------")
            print(f"训练 {target_col} 的 {s} 系模型")
            print("-------------------------------")

            df_s = df[df["Series"] == s]

            # ====== ★ 打印子集数据情况 ======
            total_n = len(df_s)
            print(f"\n【Series {s} 数据统计】")
            print(f"总样本数：    {total_n}")

            # 特征与标签
            y = df_s[target_col]
            feature_cols = [c for c in df_s.columns if c not in ["Series", "Alloy", target_col]]
            X = df_s[feature_cols]

            # 特征类型
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in X.columns if c not in num_cols]

            # 数据集划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            train_n = len(X_train)
            pct = train_n / total_n * 100

            print(f"训练样本数：  {train_n}")
            print(f"占比：        {pct:.2f}%")

            # 预处理
            if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
                OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

            preprocess = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OHE, cat_cols)
                ]
            )

            # 模型
            rf = RandomForestRegressor(n_estimators=300, random_state=42)

            model = Pipeline(steps=[
                ("preprocess", preprocess),
                ("rf", rf)
            ])

            # 训练
            model.fit(X_train, y_train)

            # 评估
            r2, mae, rmse = evaluate_model(model, X_test, y_test, f"{target_col} Series {s}")

            # 保存模型 & 成绩
            key = f"{target_col}_Series_{s}"
            all_models[key] = model
            all_scores[target_col][s] = r2

    return all_models, all_scores



# =========================================================
# ★★ 运行并输出 21 个模型中最优和最差系列 ★★
# =========================================================

models_all, scores_all = train_all_series_models_from_sheet()

print("\n================= 模型整体表现总结 =================")

targets = ["UTS", "YTS", "EL"]

# 打印表头
print(f"{'':10s} {'UTS':8s} {'YTS':8s} {'EL':8s}")

# 最佳系列
best_row = "最佳系列   "
for t in targets:
    best_s = max(scores_all[t], key=scores_all[t].get)
    best_row += f"{best_s:^8d}"
print(best_row)

# 最差系列
worst_row = "最差系列   "
for t in targets:
    worst_s = min(scores_all[t], key=scores_all[t].get)
    worst_row += f"{worst_s:^8d}"
print(worst_row)
