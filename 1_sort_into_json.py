import os
import json

# Utility function to retrieve all relevant text files from a directory.
# It removes suffixes like '-adv.txt', '-int.txt', or '-ele.txt' to get a common base name.
def get_filenames(directory, suffix):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.startswith("."):  # Skip hidden/system files
            base_name = filename.replace(suffix, "")  # Standardize the filename by removing suffix
            filenames.append((base_name, os.path.join(directory, filename)))  # Store tuple of (base_name, full_path)
    return filenames

# Reads the full content of a text file and splits it into paragraphs by empty lines.
def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()  # Trim leading/trailing whitespace
        paragraphs = content.split("\n\n")  # Paragraphs are assumed to be separated by double newlines
    return paragraphs

# Cleans input text to ensure consistency across encodings and formats.
# Includes character normalization, control character removal, punctuation fixes, and accent stripping.
def clean_text(text):
    import unicodedata
    import re

    if isinstance(text, list):
        text = " ".join(text)  # Join paragraphs into a single string if passed as list

    # Remove all ASCII control characters (e.g., \x00 to \x1F, \x7F)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # Replace common problematic Unicode symbols with ASCII equivalents
    text = text.replace("\ufeff", "")       # Byte Order Mark (BOM)
    text = text.replace("\u2013", "-")      # En dash → hyphen
    text = text.replace("\u2019", "'")      # Curly apostrophe → straight apostrophe
    text = text.replace("\u201c", '"')      # Left curly quote → double quote
    text = text.replace("\u201d", '"')      # Right curly quote → double quote
    text = text.replace("\u20ac", "EUR")    # Euro symbol → EUR

    # Normalize accented characters to plain ASCII equivalents (e.g., é → e)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

    return text.strip()  # Final trim of whitespace

# Define paths to each reading level's directory
base_dir = "OneStopEnglishCorpus/Texts-SeparatedByReadingLevel"
adv_files = get_filenames(os.path.join(base_dir, "Adv-Txt"), "-adv.txt")
int_files = get_filenames(os.path.join(base_dir, "Int-Txt"), "-int.txt")
ele_files = get_filenames(os.path.join(base_dir, "Ele-Txt"), "-ele.txt")

# Attempt to align texts from all three reading levels by matching base filenames.
# Only documents that have all three versions (advanced, intermediate, elementary) are included.
aligned_data = []
index = 1

for adv_base, adv_path in adv_files:
    int_match = next((p for b, p in int_files if b == adv_base), None)
    ele_match = next((p for b, p in ele_files if b == adv_base), None)

    if int_match and ele_match:
        aligned_data.append({
            "id": index,
            "filename": adv_base,
            "advanced": clean_text(read_text(adv_path)),
            "intermediate": clean_text(read_text(int_match)),
            "elementary": clean_text(read_text(ele_match))
        })
        index += 1

# Save the fully aligned dataset (with cleaned text) to a JSON file
output_file = "onestopenglish_aligned.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(aligned_data, f, indent=4)

print(f"✅ Successfully aligned {len(aligned_data)} documents.")
print(f"📂 Saved to {output_file}.")
