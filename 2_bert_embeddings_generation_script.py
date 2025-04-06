from sentence_transformers import SentenceTransformer
import json

# Increase timeout by setting the environment variable
import os
os.environ["HF_HUB_ENABLE_ASYNC"] = "0"  # Disables async downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # Increases timeout to 60 seconds

# Load model with cache location
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=".cache")


def get_sentence_embeddings(sentences):
    """Encodes sentences into BERT embeddings."""
    return model.encode(sentences)  # Returns a list of 384D vectors

with open("onestopenglish_aligned.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

bert_embedded_documents = []
for document in corpus:
    bert_embedded_documents.append({
            "id": document.get("id"),
            "filename": document.get("filename"),
            "advanced": get_sentence_embeddings(document.get("advanced")).tolist(),
            "intermediate": get_sentence_embeddings(document.get("intermediate")).tolist(),
            "elementary": get_sentence_embeddings(document.get("elementary")).tolist()
    })

output_file = "onestopenglish_aligned_bert_embeddings.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(bert_embedded_documents, f, indent=4)

print(f"✅ Successfully embedded {len(bert_embedded_documents)} documents.")
print(f"📂 Saved to {output_file}.")