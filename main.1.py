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
# ★★ 构造 OneHotEncoder★★
# =========================================================
def make_ohe():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:  # 新版本：用 sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:                          # 旧版本：用 sparse
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================================================
# ★★ 工具函数：根据 Alloy 提取 1–7 系 ★★
# =========================================================
def add_series_column(df):
    # Alloy 可能是数字或字符串，这里统一转为字符串后取首位
    df = df.copy()
    df["Series"] = df["Alloy"].astype(str).str[0].astype(int)
    # 只保留 1–7 系
    df = df[df["Series"].isin([1, 2, 3, 4, 5, 6, 7])]
    return df


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
# ★★ 构造一个 RF + 预处理 的 Pipeline ★★
# =========================================================
def build_rf_pipeline(num_cols, cat_cols):
    ohe = make_ohe()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols)
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf)
    ])

    return model


# =========================================================
# ★★ 针对单个 target（UTS/YTS/EL）训练 7 个专家模型 ★★
#     返回：models_target（字典），scores_target（系列→R²）
# =========================================================
def train_experts_for_target(excel_file, sheet_name, target_col):
    print("\n========================================")
    print(f"开始训练 {target_col} 的 7 个系列模型 (sheet: {sheet_name})")
    print("========================================")

    # 1. 读取并加上 Series
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = add_series_column(df)

    # 2. 全局特征与标签
    y_all = df[target_col]
    feature_cols = [c for c in df.columns if c not in ["Series", "Alloy", target_col]]
    X_all = df[feature_cols]
    series_all = df["Series"]

    # 数值 / 类别特征（对该 target 的表是固定的）
    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_all.columns if c not in num_cols]

    # 3. ★ 全局分层抽样：一次性得到 train_global / test_global ★
    (
        X_train_global,
        X_test_global,
        y_train_global,
        y_test_global,
        series_train_global,
        series_test_global
    ) = train_test_split(
        X_all,
        y_all,
        series_all,
        test_size=0.2,
        random_state=42,
        stratify=series_all
    )

    models_target = {}
    scores_target = {}

    # 4. 按 Series=1..7 训练专家模型
    for s in range(1, 8):

        print(f"\n-------------------------------")
        print(f"训练 {target_col} 的 {s} 系专家模型")
        print("-------------------------------")

        # 该系列在整张表中的总样本数
        total_n = int((series_all == s).sum())
        print(f"\n【Series {s} 数据统计】")
        print(f"总样本数：    {total_n}")

        # 训练集子集：来自 train_global 中的该系列
        train_mask = (series_train_global == s)
        X_train_s = X_train_global[train_mask]
        y_train_s = y_train_global[train_mask]

        # 测试集子集：来自 test_global 中的该系列
        test_mask = (series_test_global == s)
        X_test_s = X_test_global[test_mask]
        y_test_s = y_test_global[test_mask]

        train_n = len(X_train_s)
        test_n = len(X_test_s)

        if total_n == 0 or train_n == 0 or test_n == 0:
            print(f"⚠ Series {s} 在 {target_col} 中样本不足（total={total_n}, train={train_n}, test={test_n}），跳过该系列模型。")
            continue

        pct = train_n / total_n * 100
        print(f"训练样本数：  {train_n}")
        print(f"占比：        {pct:.2f}%")
        print(f"测试样本数：  {test_n}")

        # 5. 构建并训练该系列专家模型
        model = build_rf_pipeline(num_cols, cat_cols)
        model.fit(X_train_s, y_train_s)

        # 6. 在该系列的测试子集上评估
        r2, mae, rmse = evaluate_model(
            model, X_test_s, y_test_s,
            f"{target_col} Series {s}"
        )

        # 保存模型与 R²
        key = f"{target_col}_Series_{s}"
        models_target[key] = model
        scores_target[s] = r2

    return models_target, scores_target


# =========================================================
# ★★ 训练全部 21 个模型（UTS/YTS/EL × 7 系）★★
# =========================================================
def train_all_series_models_from_sheet():
    excel_file = "YTS UTS EL sheet.xlsx"

    configs = [
        ("UTS", "UTS"),
        ("YTS", "YTS"),
        ("EL",  "EL")
    ]

    all_models = {}
    all_scores = {"UTS": {}, "YTS": {}, "EL": {}}

    for sheet_name, target_col in configs:
        models_target, scores_target = train_experts_for_target(
            excel_file, sheet_name, target_col
        )
        # 累积结果
        all_models.update(models_target)
        all_scores[target_col] = scores_target

    return all_models, all_scores


# =========================================================
# ★★ 运行并输出 21 个模型中最优和最差系列 ★★
# =========================================================
models_all, scores_all = train_all_series_models_from_sheet()

print("\n================= 模型整体表现总结 =================")
targets = ["UTS", "YTS", "EL"]

# 表头：空白 + 三个 target
print(f"{'':10s} {'UTS':8s} {'YTS':8s} {'EL':8s}")

# 最佳系列
best_row = "最佳系列   "
for t in targets:
    if len(scores_all[t]) == 0:
        best_row += f"{'-':^8s}"
        continue
    best_s = max(scores_all[t], key=scores_all[t].get)
    best_row += f"{best_s:^8d}"
print(best_row)

# 最差系列
worst_row = "最差系列   "
for t in targets:
    if len(scores_all[t]) == 0:
        worst_row += f"{'-':^8s}"
        continue
    worst_s = min(scores_all[t], key=scores_all[t].get)
    worst_row += f"{worst_s:^8d}"
print(worst_row)
