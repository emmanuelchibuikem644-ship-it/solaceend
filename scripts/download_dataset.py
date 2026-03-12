from datasets import load_dataset
import pandas as pd

# download dataset
dataset = load_dataset("empathetic_dialogues")

# convert training split to dataframe
df = pd.DataFrame(dataset["train"])

# save as csv
df.to_csv("data/raw/empathetic_dialogues.csv", index=False)

print("Dataset downloaded and saved!")