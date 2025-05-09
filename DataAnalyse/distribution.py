import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("csv_files/final.csv")

FEATURE_COLUMNS = [
    'url_length',
    'num_special_chars',
    'digit_to_letter_ratio',
    'contains_ip',
    'primary_domain_length',
    'num_digits_primary_domain',
    'num_non_alphanumeric_primary',
    'num_hyphens_primary',
    'num_ats_primary',
    'num_dots_subdomain',
    'num_subdomains',
    'num_double_slash',
    'num_subdirectories',
    'contains_encoded_space',
    'uppercase_dirs',
    'single_char_dirs',
    'num_special_chars_path',
    'num_zeroes_path',
    'uppercase_ratio',
    'params_length',
    'num_queries'
]



TARGET_COLUMN = "target"
n_rows, n_cols = 3, 7

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), constrained_layout=True)

# for idx, feature in enumerate(FEATURE_COLUMNS):
#     row = idx // n_cols
#     col = idx % n_cols
#     ax = axes[row, col]
#     sns.boxplot(x=TARGET_COLUMN, y=feature, data=df, ax=ax)
#     ax.set_title(f"{feature}", fontsize=10)
#     ax.set_xlabel("")
#     ax.set_ylabel("")

# plt.suptitle("Boxplots Grouped by Target", fontsize=16, y=1.02)
# plt.savefig("plots/Boxplots_Grouped_by_Target.png")
# plt.show()
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), constrained_layout=True)

# for idx, feature in enumerate(FEATURE_COLUMNS):
#     row = idx // n_cols
#     col = idx % n_cols
#     ax = axes[row, col]
#     sns.violinplot(x=TARGET_COLUMN, y=feature, data=df, ax=ax)
#     ax.set_title(f"{feature}", fontsize=10)
#     ax.set_xlabel("")
#     ax.set_ylabel("")

# plt.suptitle("Violin Plots Grouped by Target", fontsize=16, y=1.02)
# plt.savefig("plots/Violin_Plots_Grouped_by_Target.png")
# plt.show()

# Set layout: 3 rows x 7 columns
n_rows = 3
n_cols = 7

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 10), constrained_layout=True)

for idx, feature in enumerate(FEATURE_COLUMNS):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]
    sns.violinplot(x=df[feature], ax=ax, inner="box", linewidth=1)
    ax.set_title(f"{feature}", fontsize=10, pad=5)
    ax.set_xlabel("")  # remove x-label if needed
plt.savefig("plots/ViolinPlots22.png")
plt.show()

# Split features
features = FEATURE_COLUMNS  # assume len(FEATURE_COLUMNS) == 21

# Set layout: 3 rows x 7 columns
n_rows = 3
n_cols = 7

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 10), constrained_layout=True)

for idx, feature in enumerate(features):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]
    sns.boxplot(x=df[feature], ax=ax)
    ax.set_title(f"{feature}", fontsize=10, pad=5)
    ax.set_xlabel("")  # remove x-label if needed

# Hide any unused subplots (not needed here since 3x7 = 21)
# fig.suptitle("Boxplots of All Features", fontsize=16, y=1.02)  # optional
plt.savefig("plots/box_plots22.png")
plt.show()