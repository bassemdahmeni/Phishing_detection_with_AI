import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("models/random_forest_pipeline2.pkl")

# Extract the RandomForest model
rf_model = pipeline.named_steps['rf']

# Your feature names list (replace this with your actual variable)
feature_names = [
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
]  # for example

# Get feature importances
importances = rf_model.feature_importances_

# Create DataFrame
feat_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_importances, x='Importance', y='Feature')
plt.title("Feature Importances from Random Forest")
plt.tight_layout()
plt.show()
