import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("csv_files/final.csv")


df1=df.groupby('target')['url_length'].mean()
print(df1)