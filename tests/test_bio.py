from cti.core.bio import load_label_map, to_bio
from cti.core.dataset import load_smah

label_map = load_label_map()
train = load_smah()["train"]
rows = train[train["entities"].notna()].head(3)

for _, row in rows.iterrows():
    print("\n", row["sentence_text"])
    for tok, s, e, tag in to_bio(row["sentence_text"], row["entities"], label_map):
        if tag != "O":
            print("  ", tok, "->", tag)