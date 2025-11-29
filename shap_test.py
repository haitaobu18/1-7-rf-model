import pandas as pd
import numpy as np
import shap
import inspect
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


# ================================
# 论文级绘图参数（全局）
# ================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11


# =========================================================
# ★★ OneHotEncoder 适配新旧版本 sklearn ★★
# =========================================================
def make_ohe():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================================================
# ★★ 根据 Alloy 提取 1–7 系 ★★
# =========================================================
def add_series_column(df):
    df = df.copy()
    df["Series"] = df["Alloy"].astype(str).str.extract(r"^(\d)").astype(float)
    df = df[df["Series"].isin([1, 2, 3, 4, 5, 6, 7])]
    df["Series"] = df["Series"].astype(int)
    return df


# =========================================================
# ★★ 训练某 1 个 target 的 7 个系列专家 + 返回 SHAP ★★
#      使用“全局统一的 OHE + StandardScaler”
# =========================================================
def train_experts_and_get_shap(excel_file, sheet_name, target_col):

    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = add_series_column(df)

    y_all = df[target_col]
    feature_cols = [c for c in df.columns if c not in ["Series", "Alloy", target_col]]
    X_all = df[feature_cols]
    series_all = df["Series"]

    # 数值 / 类别特征
    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_all.columns if c not in num_cols]

    # ---------- ★ 全局预处理器：在整张表上 fit ★ ----------
    ohe = make_ohe()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
    preprocess.fit(X_all)   # ★★ 在全体数据上统一 fit OHE，保证维度一致

    # -------------------------------
    # 分层抽样
    # -------------------------------
    (
        X_train, X_test,
        y_train, y_test,
        series_train, series_test
    ) = train_test_split(
        X_all, y_all, series_all,
        test_size=0.2,
        random_state=42,
        stratify=series_all
    )

    # 全局 transform
    X_train_pre = preprocess.transform(X_train)

    # 特征名
    num_f = list(num_cols)

    if len(cat_cols) > 0:
        try:
            ohe_f = list(ohe.get_feature_names_out(cat_cols))
        except:
            ohe_f = []
    else:
        ohe_f = []

    feature_names = num_f + ohe_f

    # -------------------------
    # SHAP 汇总容器
    # -------------------------
    shap_dict = {}
    series_train_arr = series_train.values

    # -------------------------
    # 每个 Series 训练专家模型 + SHAP
    # -------------------------
    for s in range(1, 8):

        idx_s = np.where(series_train_arr == s)[0]
        if len(idx_s) < 2:
            print(f"Series {s}: 样本不足（{len(idx_s)}），跳过。")
            continue

        X_s = X_train_pre[idx_s]
        y_s = y_train.iloc[idx_s]

        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
        rf.fit(X_s, y_s)

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_s)

        shap_dict[s] = np.abs(shap_vals).mean(axis=0)

        print(f"Series {s}: SHAP 已计算完成，样本数 = {len(idx_s)}")

    return shap_dict, feature_names


# =========================================================
# ★★ 论文级别 SHAP 分组图（Series1–3 vs Series4–7）★★
# =========================================================
def plot_grouped_shap(shap_dict, feature_names, title, save_prefix):

    os.makedirs("shap_figures", exist_ok=True)

    if not shap_dict:
        print(f"{title}: 无可用 SHAP 数据，跳过。")
        return

    series_ids = sorted(shap_dict.keys())

    # 合并 SHAP 矩阵
    shap_matrix = np.vstack([shap_dict[s] for s in series_ids])
    n_shap_features = shap_matrix.shape[1]
    n_name_features = len(feature_names)

    # 特征名补齐（处理 OHE transform 时出现的 unseen categories）
    if n_name_features < n_shap_features:
        extra_names = [f"Feature_{i}" for i in range(n_name_features, n_shap_features)]
        feature_names_extended = feature_names + extra_names
    else:
        feature_names_extended = feature_names

    # Top K 特征
    top_k = min(12, n_shap_features)
    mean_importance = shap_matrix.mean(axis=0)
    top_idx = np.argsort(mean_importance)[-top_k:][::-1]
    top_features = [feature_names_extended[i] for i in top_idx]

    # Series 1–3 vs 4–7
    groupA_rows = [shap_dict[s] for s in series_ids if s in (1, 2, 3)]
    groupB_rows = [shap_dict[s] for s in series_ids if s in (4, 5, 6, 7)]

    group_A = np.vstack(groupA_rows).mean(axis=0)[top_idx] if groupA_rows else np.zeros(top_k)
    group_B = np.vstack(groupB_rows).mean(axis=0)[top_idx] if groupB_rows else np.zeros(top_k)

    # 绘图
    y = np.arange(len(top_features))
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(y - 0.18, group_A, height=0.35,
            color="#D62728", label="Series 1–3", alpha=0.9)
    ax.barh(y + 0.18, group_B, height=0.35,
            color="#FF9896", label="Series 4–7", alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"shap_figures/{save_prefix}.png", dpi=300)
    plt.savefig(f"shap_figures/{save_prefix}.pdf")
    plt.close()

    print(f"✔ 已生成论文级图像：shap_figures/{save_prefix}.png")


# =========================================================
# ★★ 主流程 — 训练三大类并绘图 ★★
# =========================================================
if __name__ == "__main__":
    excel_file = "YTS UTS EL sheet.xlsx"

    targets = [
        ("UTS", "UTS"),
        ("YTS", "YTS"),
        ("EL",  "EL")
    ]

    for sheet_name, target in targets:
        print(f"\n===== 开始处理 {target} =====")

        shap_dict, feature_names = train_experts_and_get_shap(
            excel_file, sheet_name, target
        )

        plot_grouped_shap(
            shap_dict,
            feature_names,
            f"{target} — SHAP Grouped Importance",
            f"{target}_SHAP_grouped"
        )

    print("\n🎉 所有 SHAP 图已经生成完毕（3 张 PNG + 3 张 PDF）")
