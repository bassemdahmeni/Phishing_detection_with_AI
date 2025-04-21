import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("csv_files/final.csv")

# Quick glimpse of the dataset
# print("Shape of the dataset:", df.shape)
# print("First 5 rows:\n", df.head())
# print("Data Types:\n", df.info())
# print("Summary Statistics:\n", df.describe())
# # Check for missing values
# print("Missing values:\n", df.isnull().sum())

# # Check for duplicate rows
# duplicates = df.duplicated().sum()
# print(f"Number of duplicate rows: {duplicates}")
# Drop unnecessary columns
cols_to_drop = ["id", "url", "url_hash", "top_level_domain", "primary_domain", "created_at"]
df = df.drop(columns=cols_to_drop)
# Define your feature columns (adjust as needed)
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

# Plot Histograms for all features
df[FEATURE_COLUMNS].hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Features")
plt.tight_layout()
plt.show()
#Boxplots & Violin Plots
for feature in FEATURE_COLUMNS:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.subplot(1, 2, 2)
    sns.violinplot(x=df[feature])
    plt.title(f"Violin Plot of {feature}")
    plt.tight_layout()
    plt.show()



# Compute correlation matrix
corr = df[FEATURE_COLUMNS].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()



TARGET_COLUMN = "target"

for feature in FEATURE_COLUMNS:
    plt.figure(figsize=(12, 4))

    #Boxplot grouped by target
    plt.subplot(1, 2, 1)
    sns.boxplot(x=TARGET_COLUMN, y=feature, data=df)
    plt.title(f"Boxplot of {feature} by {TARGET_COLUMN}")

    #Violin plot grouped by target
    plt.subplot(1, 2, 2)
    sns.violinplot(x=TARGET_COLUMN, y=feature, data=df)
    plt.title(f"Violin Plot of {feature} by {TARGET_COLUMN}")

    plt.tight_layout()
    plt.show()



sns.pairplot(df[FEATURE_COLUMNS + [TARGET_COLUMN]], hue=TARGET_COLUMN)
plt.suptitle("Pairplot of Features Colored by Target", y=1.02)
plt.show()




