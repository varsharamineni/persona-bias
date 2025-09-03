import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pandas as pd

# Paths to JSONL files
files = {
    "Trump": Path("results/mmlu/Qwen/Qwen3-4B/adopt_identity_accordance/college_biology/testrun_a_Donald_Trump_supporter_text_predictions_labeled.jsonl"),
    "AbleBodied": Path("results/mmlu/Qwen/Qwen3-4B/adopt_identity_accordance/college_biology/testrun_an_able-bodied_person_text_predictions_labeled.jsonl"),
    "AfricanPerson": Path("results/mmlu/Qwen/Qwen3-4B/adopt_identity_accordance/college_biology/testrun_an_African_person_text_predictions_labeled.jsonl"),
}

# Load JSONL files
data = {}
for key, path in files.items():
    with open(path, "r", encoding="utf-8") as f:
        data[key] = {json.loads(line)["id"]: json.loads(line)["predicted_explanations"] for line in f}

# Make list of ids (assuming all files have same ids)
ids = list(data["Trump"].keys())

# Function to compute TF-IDF cosine similarity
def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Function to compute character-level similarity
def char_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()

# Prepare dataframe for similarity results
results = []

for i in ids:
    expl_trump = data["Trump"][i]
    expl_able = data["AbleBodied"][i]
    expl_african = data["AfricanPerson"][i]

    sim_trump_able = tfidf_similarity(expl_trump, expl_able)
    sim_trump_african = tfidf_similarity(expl_trump, expl_african)
    sim_able_african = tfidf_similarity(expl_able, expl_african)

    char_trump_able = char_similarity(expl_trump, expl_able)
    char_trump_african = char_similarity(expl_trump, expl_african)
    char_able_african = char_similarity(expl_able, expl_african)

    results.append({
        "id": i,
        "tfidf_trump_able": sim_trump_able,
        "tfidf_trump_african": sim_trump_african,
        "tfidf_able_african": sim_able_african,
        "char_trump_able": char_trump_able,
        "char_trump_african": char_trump_african,
        "char_able_african": char_able_african,
    })

df = pd.DataFrame(results)

# Summary statistics
summary = df.describe()
print(summary)