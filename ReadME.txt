==========================================

Project Title: Text Simplification Using Graph Attention Networks on BERT Embeddings
Course: SCC.413 Natural Language Processing

This project implements a text simplification pipeline that transforms BERT embeddings of complex texts into simplified equivalents using Graph Attention Networks (GATs). It includes data preprocessing, embedding extraction, model training, inference, and evaluation.

----------------------------------------------------

REQUIREMENTS
----------------------------------------------------
This project requires the following libraries:
- Python 3.8+
- PyTorch
- PyTorch Geometric
- Transformers (HuggingFace)
- scikit-learn
- pandas
- matplotlib
- tqdm
- numpy

To install dependencies:
pip install -r requirements.txt

----------------------------------------------------

FILE OVERVIEW
----------------------------------------------------
1_sort_into_json.py

Purpose:
Parses the OneStopEnglish dataset and aligns texts at different simplification levels (advanced, intermediate, elementary) into a structured JSON format.

How to run:
python 1_sort_into_json.py

Output:
onestopenglish_alligned.json: Contains aligned triplets of texts at three reading levels.


2_bert_embeddings_generation_script.py

Purpose:
Generates BERT embeddings (384-dimensional) for each text using Sentence-BERT and saves them along with their filenames.

How to run:
python 2_bert_embeddings_generation_script.py


Output:
onestopenglish_alligned_bert_embeddings.json

Each JSON file contains id, filename and BERT embeddings at the 3 different levels of english.


3_gat_train_evaluate_model.py

Purpose:
Trains a GAT model to map advanced BERT embeddings to both intermediate or elementary embeddings. Supports cosine-based graph construction.

How to run:
python 3_gat_train_evaluate_model.py

Output:
Model training logs
Loss curve plots: loss_curve_elementary.png, loss_curve_intermediate.png
Trained model weights


4_text_simplification_inference.py

Purpose:
Applies the trained GAT model to transform test embeddings and evaluates the cosine similarity to ground-truth simplified embeddings.

How to run:
python 4_text_simplification_inference.py --input onestopenglish_aligned_bert_embeddings.json --output results_directory
Mandatory arguments:
--input: Filepath of the BERT embeddings dataset (default: filename suficies)
--output: Directory for saving results (default: results_directory)

The evaluation script will:

Load the trained models
Apply them to transform advanced embeddings
Calculate similarity metrics with ground truth embeddings
Generate visualizations
Save detailed results to the output directory


5_simplification_nearest_neighbour.py

Purpose:
Implements the decoding step by matching each predicted simplified embedding with the nearest real simplified embedding from the dataset.

How to run:
python 5_simplification_nearest_neighbour.py

Output:
Nearest neighbor predictions: saved as JSON for qualitative analysis in 'paired_output.json'.

----------------------------------------------------

HOW TO REPRODUCE RESULTS
----------------------------------------------------

1. Prepare the dataset using 1_sort_into_json.py

2. Generate BERT embeddings using 2_bert_embeddings_generation_script.py

3. Train the GAT model using 3_gat_train_evaluate_model.py

4. Run inference and evaluation with 4_text_simplification_inference.py

5. Decode embeddings using nearest neighbor retrieval via 5_simplification_nearest_neighbour.py

6. Refer to the output plots and metrics files for analysis and reporting.