# -----------------------------------------------Normalized-----------------------------------------------------------------------------------------
import pandas as pd

# Function to perform min-max normalization
def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())

# Load your datasets
set_a = pd.read_csv('NSR.csv')
# set_b = pd.read_csv('SetB_MI.csv')
# set_c = pd.read_csv('SetC_MI.csv')

# Apply the normalization function to each dataset
set_a_normalized = min_max_normalize(set_a)
# set_b_normalized = min_max_normalize(set_b)
# set_c_normalized = min_max_normalize(set_c)

# You can view the first few rows of the normalized data to check the results
print(set_a_normalized.head())
# print(set_b_normalized.head())
# print(set_c_normalized.head())

# Save the normalized datasets to CSV files
set_a_normalized.to_csv('NSR_Normalized.csv', index=False)
# set_b_normalized.to_csv('SetB_MI_Normalized.csv', index=False)
# set_c_normalized.to_csv('SetC_MI_Normalized.csv', index=False)
