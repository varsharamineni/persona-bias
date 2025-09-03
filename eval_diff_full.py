import json
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pandas as pd

# Folder containing the JSONL files
folder_path = Path("results/mmlu/Qwen/Qwen3-8B/adopt_identity_accordance/college_biology")

# Load all matching JSONL files
data = {}
for file in folder_path.glob("testrun_*_text_predictions_labeled.jsonl"):
    # Clean identity name: remove 'testrun_' prefix and '_text_predictions_labeled' suffix
    identity_name = file.stem.replace("testrun_", "").replace("_text_predictions_labeled", "")
    
    with open(file, "r", encoding="utf-8") as f:
        data[identity_name] = {
            json.loads(line)["id"]: json.loads(line)["predicted_explanations"]
            for line in f
        }

# List of question IDs (assumes all files have the same set)
ids = list(next(iter(data.values())).keys())

# TF-IDF cosine similarity function
def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Character-level similarity function
def char_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()

# Compute pairwise similarities for all identity pairs
results = []
identity_pairs = list(combinations(data.keys(), 2))

for i in ids:
    row = {"id": i}
    for id1, id2 in identity_pairs:
        expl1 = data[id1][i]
        expl2 = data[id2][i]

        row[f"tfidf_{id1}_{id2}"] = tfidf_similarity(expl1, expl2)
        row[f"char_{id1}_{id2}"] = char_similarity(expl1, expl2)

    results.append(row)

df = pd.DataFrame(results)

# Print summary statistics
summary = df.describe()
print(summary)

# Save full results to CSV
df.to_csv("similarity_results.csv", index=False)
