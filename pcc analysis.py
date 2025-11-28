import matplotlib
matplotlib.use("TkAgg")   # Fix PyCharm backend issue

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =============================
# 1. Load Excel File
# =============================
path = "YTS UTS EL sheet.xlsx"
save_dir = os.path.dirname(os.path.abspath(__file__))

xlsx = pd.ExcelFile(path)

# ★★★ 正确匹配 sheet → 目标列 ★★★
sheet_names_map = {
    "YTS": xlsx.sheet_names[0],   # 第一张表是 YTS
    "UTS": xlsx.sheet_names[1],   # 第二张表是 UTS
    "EL":  xlsx.sheet_names[2]    # 第三张表是 EL
}


# =============================
# 自动识别真实目标列名
# =============================
def find_target_column(columns, target_key):
    target_key = target_key.lower()
    for col in columns:
        col_low = col.lower()
        if target_key == "uts" and "uts" in col_low:
            return col
        if target_key == "yts" and ("yts" in col_low or "ys" in col_low):
            return col
        if target_key == "el" and ("el" in col_low or "elong" in col_low):
            return col
    return None


# =============================
# 2. Heatmap With Numerical Annotations
# =============================
def save_corr_heatmap_with_numbers(df, title, filename):
    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr(method='pearson')

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Pearson Correlation Heatmap — {title}", fontsize=18)
    plt.colorbar()

    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90, fontsize=10)
    plt.yticks(ticks, corr.columns, fontsize=10)

    # annotate value in each cell
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            plt.text(j, i, f"{value:.2f}", ha='center', va='center',
                     color=text_color, fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[图像已保存] {save_path}")


# =============================
# 3. Compute Key & Noise Features
# =============================
def analyze_key_features(df, target_col):
    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr()

    if target_col not in corr.columns:
        print(f"[警告] {target_col} 不在 corr() 矩阵中，可能不是数值列")
        return None

    s = corr[target_col].drop(target_col)

    positive = s.sort_values(ascending=False).head(5)
    negative = s.sort_values().head(5)
    noise = s.abs().sort_values().head(5)

    return positive, negative, noise


# =============================
# 新增功能：生成 Pearson 相关矩阵字典（用于打印与导出 Excel）
# =============================
corr_matrix_dict = {}   # 保存三个 sheet 的皮尔逊矩阵


# =============================
# 4. MAIN LOOP
# =============================
for target_key, sheet_name in sheet_names_map.items():
    print("\n" + "=" * 80)
    print(f"📌 {target_key} — 关键特征分析")
    print("=" * 80)

    df = xlsx.parse(sheet_name)

    # --- 自动识别列名 ---
    target_col = find_target_column(df.columns, target_key)
    if not target_col:
        print(f"[错误] 在 sheet '{sheet_name}' 中找不到 {target_key} 列")
        print("列名如下：")
        print(list(df.columns))
        continue

    print(f"[匹配到的目标列] {target_col}")

    # --- 强制转成数值（重要） ---
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # --- 计算相关性（整个矩阵） ---
    num_df = df.select_dtypes(include=['number'])
    corr_full = num_df.corr(method='pearson')

    # 保存到字典方便导出
    corr_matrix_dict[target_key] = corr_full

    # --- 计算关键特征 ---
    result = analyze_key_features(df, target_col)

    if result:
        positive, negative, noise = result

        print("\n[正相关关键特征]")
        print(positive.to_string())

        print("\n——————")

        print("\n[负相关关键特征]")
        print(negative.to_string())

        print("\n——————")

        print("\n[噪声特征（最弱相关）]")
        print(noise.to_string())
        print("\n")

    # --- 保存热力图 ---
    filename = f"{target_key}_corr_heatmap_annotated.png"
    save_corr_heatmap_with_numbers(df, f"{target_key} Sheet", filename)


# =============================
# 5. 新增功能：在结果框打印三个 Pearson 矩阵
# =============================
print("\n" + "=" * 80)
print("📌 全部 Pearson 相关性矩阵")
print("=" * 80)

for key, mat in corr_matrix_dict.items():
    print(f"\n===== {key} Pearson Correlation Matrix =====\n")
    print(mat.round(4))


# =============================
# 6. 新增功能：导出 Pearson matrix.xlsx（含 3 个 sheet）
# =============================
output_path = os.path.join(save_dir, "Pearson matrix.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for key, mat in corr_matrix_dict.items():
        mat.to_excel(writer, sheet_name=key)

print(f"\n🎉 Pearson matrix.xlsx 已成功生成： {output_path}")
print("📌 文件包含 3 个 sheet：YTS、UTS、EL")
print("📌 全部分析完成！")
