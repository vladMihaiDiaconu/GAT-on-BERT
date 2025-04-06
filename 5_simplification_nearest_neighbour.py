import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load GAT output
with open("results_directory/simplification_results_elementary.json", "r") as f:
    gat_data = json.load(f)

# Load OneStopEnglish aligned reference embeddings
with open("onestopenglish_aligned_bert_embeddings.json", "r") as f:
    reference_data = json.load(f)

# Prepare ground-truth embedding matrices and filenames
elementary_embeddings = np.array([item["elementary"] for item in reference_data])
advanced_embeddings = np.array([item["advanced"] for item in reference_data])
filenames = [item["filename"] for item in reference_data]

# Prepare the result list
final_pairs = []

# Match each GAT output to the closest ground-truth simplified and advanced embeddings
for entry in tqdm(gat_data, desc="Pairing embeddings"):
    simplified_emb = np.array(entry["simplified_embedding"]).reshape(1, -1)

    # Cosine similarity search
    elem_sims = cosine_similarity(simplified_emb, elementary_embeddings)[0]
    adv_sims = cosine_similarity(simplified_emb, advanced_embeddings)[0]

    # Get indices of best matches
    best_elem_idx = np.argmax(elem_sims)
    best_adv_idx = np.argmax(adv_sims)

    final_pairs.append({
        "id": entry["id"],
        "source_filename": entry["filename"],
        "matched_elementary_filename": filenames[best_elem_idx],
        "matched_advanced_filename": filenames[best_adv_idx],
        "elementary_similarity": float(elem_sims[best_elem_idx]),
        "advanced_similarity": float(adv_sims[best_adv_idx])
    })

# Save final result
with open("paired_output.json", "w") as f:
    json.dump(final_pairs, f, indent=2)

print("Done! Output saved to 'paired_output.json'.")
