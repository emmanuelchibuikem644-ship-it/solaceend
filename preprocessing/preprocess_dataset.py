import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/empathetic_dialogues.csv"
PROCESSED_PATH = "data/processed/processed_dataset.csv"


def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()

    return text


def preprocess():

    df = pd.read_csv(RAW_PATH)

    df = df[["utterance", "context"]]

    df = df.rename(columns={
        "utterance": "text",
        "context": "emotion"
    })

    df["text"] = df["text"].apply(clean_text)

    encoder = LabelEncoder()

    df["emotion"] = encoder.fit_transform(df["emotion"])

    df.to_csv(PROCESSED_PATH, index=False)

    return df


def split_dataset(df):

    train, temp = train_test_split(df, test_size=0.2, random_state=42)

    val, test = train_test_split(temp, test_size=0.5)

    train.to_csv("data/splits/train.csv", index=False)
    val.to_csv("data/splits/validation.csv", index=False)
    test.to_csv("data/splits/test.csv", index=False)


if __name__ == "__main__":

    dataset = preprocess()

    split_dataset(dataset)

    print("Preprocessing complete.")