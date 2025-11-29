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
# OneHotEncoder 适配新旧 sklearn
# =========================================================
def make_ohe():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================================================
# 根据 Alloy 提取 1–7 系
# =========================================================
def add_series_column(df):
    df = df.copy()
    df["Series"] = df["Alloy"].astype(str).str.extract(r"^(\d)").astype(float)
    df = df[df["Series"].isin([1, 2, 3, 4, 5, 6, 7])]
    df["Series"] = df["Series"].astype(int)
    return df


# =========================================================
# 第一部分：7个专家模型 → series-level mean(|SHAP|)
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

    # ---------- 全局 OHE + 标准化 ----------
    ohe = make_ohe()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
    preprocess.fit(X_all)

    # ---------- 分层抽样 ----------
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

    # ---------- 逐系列计算专家 SHAP ----------
    shap_dict = {}
    series_train_arr = series_train.values

    for s in range(1, 8):

        idx_s = np.where(series_train_arr == s)[0]
        if len(idx_s) < 2:
            print(f"Series {s}: 样本不足（{len(idx_s)}），跳过。")
            continue

        X_s = X_train_pre[idx_s]
        y_s = y_train.iloc[idx_s]

        # 随机森林专家
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
        rf.fit(X_s, y_s)

        # SHAP（取绝对值平均）
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_s)

        shap_dict[s] = np.abs(shap_vals).mean(axis=0)

        print(f"Series {s}: SHAP 已计算完成，样本数 = {len(idx_s)}")

    return shap_dict, feature_names


# =========================================================
# 7 系列的 beeswarm（你原本用来替代柱状图的版本）
# =========================================================
def plot_beeswarm(shap_dict, feature_names, title, save_prefix):

    os.makedirs("shap_figures", exist_ok=True)

    if not shap_dict:
        print(f"{title}: 无 SHAP 数据，跳过。")
        return

    # 合并所有 series 的 SHAP，形成 matrix
    shap_matrix = np.vstack([shap_dict[s] for s in sorted(shap_dict.keys())])
    n_features = shap_matrix.shape[1]

    # 特征名补齐
    if len(feature_names) < n_features:
        extra = [f"Feature_{i}" for i in range(len(feature_names), n_features)]
        feature_names = feature_names + extra

    # 选取 top20 特征
    top_k = min(20, n_features)
    top_idx = np.argsort(np.mean(shap_matrix, axis=0))[-top_k:][::-1]

    shap_top = shap_matrix[:, top_idx]
    feature_top_names = [feature_names[i] for i in top_idx]

    # ------------------------ 自制 beeswarm -------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(top_k):
        sv = shap_top[:, i]
        y = np.random.normal(i, 0.12, size=len(sv))
        ax.scatter(sv, y, s=18, alpha=0.6, color="#1f77b4")

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feature_top_names)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(f"shap_figures/{save_prefix}.png", dpi=300)
    fig.savefig(f"shap_figures/{save_prefix}.pdf")
    plt.close()

    print(f"✔ Series-level Beeswarm SHAP 图已生成：shap_figures/{save_prefix}.png")


# =========================================================
# 第二部分：真正的 summary plot（逐样本 + 正负贡献）
# =========================================================
def train_global_shap(excel_file, sheet_name, target_col):

    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = add_series_column(df)

    y = df[target_col]
    X = df[[c for c in df.columns if c not in ["Series", "Alloy", target_col]]]

    # 数值 / 类别特征
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # ---------- 统一预处理 ----------
    ohe = make_ohe()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )

    # fit + transform
    X_trans = preprocess.fit_transform(X)

    # ---------- 关键修复：使用“fit 之后”的 OHE ----------
    ohe_fitted = preprocess.named_transformers_["cat"]

    # 获取特征名
    num_f = num_cols
    if len(cat_cols) > 0:
        try:
            ohe_f = list(ohe_fitted.get_feature_names_out(cat_cols))
        except:
            ohe_f = []
    else:
        ohe_f = []

    feature_names = num_f + ohe_f

    # ---------- 全局模型 ----------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_trans, y)

    # ---------- SHAP ----------
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_trans)

    return shap_values, X_trans, feature_names



# =========================================================
# summary plot（你要的全局 SHAP 类型：正负贡献 + 全局重要性）
# =========================================================
def plot_shap_summary(shap_values, X_trans, feature_names, title, save_prefix):

    os.makedirs("shap_figures", exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        plot_type="dot",
        max_display=20,
        show=False,
        color_bar=False     # 避免 matplotlib 3.8 的 colorbar 错误
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"shap_figures/{save_prefix}.png", dpi=300)
    plt.savefig(f"shap_figures/{save_prefix}.pdf")
    plt.close()

    print(f"✔ Summary SHAP 图已生成：shap_figures/{save_prefix}.png")


# =========================================================
# 主流程
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

        # 第一部分：series-level SHAP（你的原始逻辑）
        shap_dict, feature_names = train_experts_and_get_shap(
            excel_file, sheet_name, target
        )
        plot_beeswarm(
            shap_dict,
            feature_names,
            f"{target} — Series-level Beeswarm SHAP",
            f"{target}_SHAP_beeswarm"
        )

        # 第二部分：summary plot（你要的真正 SHAP 形式）
        shap_vals, X_trans, fnames = train_global_shap(
            excel_file, sheet_name, target
        )
        plot_shap_summary(
            shap_vals, X_trans, fnames,
            f"{target} — SHAP Summary Plot",
            f"{target}_SHAP_summary"
        )

    print("\n🎉 所有图像已生成（Beeswarm + Summary）")
