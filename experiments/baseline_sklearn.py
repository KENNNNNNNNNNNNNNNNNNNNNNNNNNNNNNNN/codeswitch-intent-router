from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

data = pd.read_csv("data/dataset_with_density.csv")
splits = pd.read_csv("data/splits.csv")[["id", "split"]]
df = data.drop(columns=["split"]).merge(splits, on="id")

# Character n-grams cope better than whole words with mixed Tagalog/English spelling.
model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
    LogisticRegression(max_iter=1000),
)
train = df[df.split == "train"]
model.fit(train["text"], train["intent"])


def bucket(d):
    if d == 0:
        return "1 single-language (0)"
    if d <= 0.3:
        return "2 low-mixed (0-0.3]"
    return "3 high-mixed (>0.3)"


df["bucket"] = df["density"].apply(bucket)
labels = sorted(df["intent"].unique())
print("Intent order for the confusion matrix:")
for i, name in enumerate(labels):
    print(f"  {i} = {name}")
print()

saved = []
for split_name in ["validation", "test"]:
    part = df[df.split == split_name].copy()
    part["pred"] = model.predict(part["text"])
    print(f"=== {split_name} (n={len(part)}) ===")
    print("accuracy:", round(accuracy_score(part.intent, part.pred), 3))
    print("macro F1:", round(f1_score(part.intent, part.pred, average="macro", zero_division=0), 3))
    matrix = confusion_matrix(part.intent, part.pred, labels=labels)
    print("confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(matrix))
    print("by density bucket:")
    for name, group in part.groupby("bucket"):
        acc = accuracy_score(group.intent, group.pred)
        print(f"  {name}: n={len(group)}, accuracy={acc:.2f}")
    print()
    saved.append(part[["id", "text", "intent", "density", "bucket", "pred"]].assign(split=split_name))

# Save predictions so later stages can compare models sentence by sentence.
Path("results").mkdir(exist_ok=True)
pd.concat(saved).to_csv("results/baseline_predictions.csv", index=False)
print("saved predictions to results/baseline_predictions.csv")