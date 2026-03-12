import pandas as pd
from sklearn.model_selection import train_test_split

# load processed dataset
df = pd.read_csv("data/processed/processed_dataset.csv")

# split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# save splits
train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/validation.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

print("Dataset splitting complete.")