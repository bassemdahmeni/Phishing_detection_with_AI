import pandas as pd

def load_first_rows(csv_file, num_rows=209184):
    """
    Loads the first `num_rows` rows from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.
        num_rows (int): Number of rows to load (default is 209184).

    Returns:
        pandas.DataFrame: DataFrame containing the first `num_rows` rows.
    """
    df = pd.read_csv(csv_file, nrows=num_rows)
    return df

# Example usage:
df_first_rows = load_first_rows('csv_files/top2.csv')
df_first_rows.to_csv('csv_files/legitimate.csv', index=False)
