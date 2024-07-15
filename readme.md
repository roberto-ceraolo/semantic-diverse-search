# Sentence Pair Finder

## Overview
The Sentence Pair Finder is a sophisticated tool designed to identify semantically similar sentence pairs within and across languages, specifically focusing on English and German. This project leverages state-of-the-art natural language processing techniques to find diverse and meaningful sentence pairs in large legal datasets. Additionally, it includes a fine-tuning procedure for the `joelniklaus/legal-swiss-longformer-base` model using a self-supervised learning approach.

## Features
- Efficient processing of large-scale legal datasets
- Monolingual pair finding for English and German
- Cross-lingual pair finding between English and German
- Semantic similarity calculation using advanced embedding techniques
- Lexical diversity measurement for enhanced pair selection
- Customizable configuration through YAML files
- Caching mechanism for faster processing
- Comprehensive evaluation metrics including BLEU, BERT Score, and embedding similarity
- Optional GPT-based evaluation for enhanced semantic analysis
- CSV output generation for easy result analysis
- Streamlit dashboard for interactive result visualization
- Fine-tuning procedure for the `joelniklaus/legal-swiss-longformer-base` model

## Requirements
- Python 3.7+
- PyTorch
- Sentence Transformers
- Hugging Face Datasets
- FAISS
- NLTK
- scikit-learn
- OpenAI API (optional, for GPT-based evaluation)
- Streamlit (for dashboard)

## Installation
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up the OpenAI API key (if using GPT-based evaluation):
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Configuration
Modify the `config.yaml` file to customize the behavior of the Sentence Pair Finder and the fine-tuning process:

```yaml
# Sentence Pair Finder Configuration
dataset_name: "joelniklaus/Multi_Legal_Pile"
en_subset: "en_contracts"
de_subset: "de_contracts"
transformer_model: "paraphrase-multilingual-MiniLM-L12-v2"
num_samples: 20000
buffer_size: 20000
min_sentence_length: 5
max_sentence_length: 50
max_sentences: 50000
k: 1000
num_query_samples: 100
semantic_threshold: 0.9
cross_lingual_threshold: 0.9
toggle_GPT: false
num_eval_samples: 100
gpt_model: "gpt-3.5-turbo"
openai_api_key: "your-api-key-here"  # Replace with your actual API key
generate_csv: true  # Set to true to generate CSV files
csv_output_dir: "output"  # Directory to store CSV files
output_dir: "output"

# Fine-Tuning Configuration
fine_tuning:
  model_name: 'joelniklaus/legal-swiss-longformer-base'
  dataset_name: 'joelniklaus/Multi_Legal_Pile'
  output_dir: 'fine_tuned_model'
  batch_size: 32
  num_epochs: 5
  learning_rate: 2e-5
```

## Usage
1. Run the main script for sentence pair finding:
   ```
   python main.py
   ```

2. View the results:
   - Check the generated log file in the `logs` directory for detailed evaluation results.
   - If CSV generation is enabled, find the output files in the specified `csv_output_dir`.

3. Launch the Streamlit dashboard:
   ```
   streamlit run streamlit-dashboard.py
   ```

4. Fine-tune the model:
   ```
   python -m dataset
   python main.py
   ```

## Project Structure
- `main.py`: Entry point of the application
- `pair_finder.py`: Core logic for finding sentence pairs
- `evaluator.py`: Evaluation metrics calculation
- `utils.py`: Utility functions and configuration management
- `streamlit-dashboard.py`: Interactive dashboard for result visualization
- `fine_tuning`: A folder containing:
  - `dataset.py`: Dataset preparation for fine-tuning
  - `model.py`: Model definition for fine-tuning
  - `train.py`: Training script for fine-tuning
  - `config.yaml`: Configuration file for the Sentence Pair Finder and fine-tuning process
- `experiment_data`: A folder containing some sample experiment data, to be viewed in the Streamlit dashboard

## Evaluation Metrics
The project uses several metrics to evaluate the quality of found pairs:
- BLEU Score: Measures the quality of machine-translated text
- BERT Score: Assesses semantic similarity using BERT embeddings
- Embedding Similarity: Cosine similarity of sentence embeddings
- Lexical Diversity: Measures the vocabulary richness of paired sentences
- GPT-based Evaluation (optional): Utilizes GPT models for semantic analysis

## Dashboard
The Streamlit dashboard provides an interactive interface to:
- View experiment configurations
- Analyze evaluation results across different languages
- Visualize metrics through charts and graphs
- Explore example sentence pairs

## Fine-Tuning Procedure

### Overview
The project includes a procedure for fine-tuning the `joelniklaus/legal-swiss-longformer-base` model using a self-supervised learning approach. This approach leverages the `joelniklaus/Multi_Legal_Pile` dataset to generate sentence embeddings and create similarity-based triplets for contrastive learning.

### Steps Involved:
1. **Dataset Preparation**:
   - Collect samples from the `joelniklaus/Multi_Legal_Pile` dataset
   - Compute embeddings using the pre-trained model
   - Generate a similarity matrix for embeddings
   - Save collected sentences and similarity matrix

2. **Fine-Tuning**:
   - Load the prepared dataset
   - Create triplets based on similarity scores
   - Train the model using triplet margin loss

### Implementation Details:
The fine-tuning process is organized into several key components:

1. **Configuration**: `config.yaml`
2. **Dataset Preparation**: `dataset.py`
   - Class: `SimilarityBasedTripletDataset`
   - Key functions: `_get_embeddings`, `create_and_save_triplet_dataset`, `load_triplet_dataset`
3. **Model Definition**: `model.py`
   - Class: `TripletLossModel`
4. **Training Script**: `train.py`
   - Function: `train`

### Instructions for Fine-Tuning:
1. Prepare the environment and set up configuration parameters in `config.yaml`.
2. Fine-tune the model:
   ```sh
   python main.py
   ```

This fine-tuning procedure enhances the model's performance on legal domain tasks, improving the overall quality of sentence pair finding and similarity assessments.