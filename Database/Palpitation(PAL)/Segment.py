import pandas as pd

# Load the dataset
df = pd.read_csv("PAL_Normalized.csv")

# Segment the dataset
set_a = df.iloc[0:1000]
set_b = df.iloc[1000:3000]
set_c = df.iloc[3000:6000]

# Save each set to separate CSV files
set_a.to_csv("SetA_PAL_Normalized.csv", index=False)
set_b.to_csv("SetB_PAL_Normalized.csv", index=False)
set_c.to_csv("SetC_PAL_Normalized.csv", index=False)

print("Segmentation complete. Files saved as SetA.csv, SetB.csv, and SetC.csv.")
