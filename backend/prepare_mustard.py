import pandas as pd
import json

# Load Excel dataset
df = pd.read_excel("data/mustard_sarcasm_dataset.xlsx")

print("Original rows:", len(df))

# Convert columns
new_df = pd.DataFrame()

new_df["scene"] = ["mustard_" + str(i + 1) for i in range(len(df))]
new_df["key"] = df["key"]

# Context
def clean_context(value):
    if isinstance(value, list):
        return " ".join(value)

    value = str(value)

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return " ".join(parsed)
    except:
        pass

    return value

new_df["context"] = df["context"].apply(clean_context)

# Target response
new_df["response"] = df["utterance"]

# Sarcasm label
new_df["label"] = df["sarcasm"].astype(int)

# Metadata
new_df["sarcasm_type"] = "unknown"
new_df["speaker"] = df["speaker"]
new_df["show"] = df["show"]

# Fields not available in MUStARD
new_df["implicit_emotion"] = "unknown"
new_df["explicit_emotion"] = "unknown"
new_df["valence"] = "unknown"
new_df["arousal"] = "unknown"

# Correct column order
columns = [
    "scene",
    "key",
    "context",
    "response",
    "label",
    "sarcasm_type",
    "speaker",
    "show",
    "implicit_emotion",
    "explicit_emotion",
    "valence",
    "arousal"
]

new_df = new_df[columns]

# Remove invalid rows
new_df = new_df.dropna(
    subset=["context", "response", "label"]
)

new_df = new_df[
    (new_df["context"].str.strip() != "") &
    (new_df["response"].str.strip() != "")
]

# Save
output = "data/mustard_ready.csv"

new_df.to_csv(output, index=False)

print("\nConverted successfully!")
print("Rows:", len(new_df))
print("\nLabels:")
print(new_df["label"].value_counts())

print("\nSaved to:", output)