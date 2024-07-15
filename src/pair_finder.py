import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import random
from collections import deque
import numpy as np
import faiss
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json
from typing import List, Tuple, Dict, Any
from utils import Config, setup_logging
import csv
import os
import hashlib
import pickle

class SentencePairFinder:
    def __init__(self, config: Config):
        self.config = config
        self.sentence_transformer = SentenceTransformer(config.transformer_model)
        nltk.download('punkt')
        nltk.download('stopwords')
        self.english_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.german_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        self.stop_words = set(stopwords.words('english'))
        self.cache_dir = os.path.join(config.output_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self) -> str:
        """Generate a unique cache key based on relevant configuration settings."""
        relevant_config = {
            'dataset_name': self.config.dataset_name,
            'en_subset': self.config.en_subset,
            'de_subset': self.config.de_subset,
            'num_samples': self.config.num_samples,
            'buffer_size': self.config.buffer_size,
            'transformer_model': self.config.transformer_model,
            'max_sentences': self.config.max_sentences,
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_embeddings(self, embeddings: torch.Tensor, sentences: List[str], lang: str):
        """
        Save the computed embeddings to a file.
        """
        cache_key = self.get_cache_key()
        file_path = os.path.join(self.cache_dir, f'{cache_key}_{lang}_embeddings.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'sentences': sentences}, f)
        logging.info(f"Saved {lang} embeddings to {file_path}")

    def load_embeddings(self, lang: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Load the embeddings from a file.
        """

        cache_key = self.get_cache_key()
        file_path = os.path.join(self.cache_dir, f'{cache_key}_{lang}_embeddings.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded {lang} embeddings from {file_path}")
            return data['embeddings'], data['sentences']
        return None, None

    def save_faiss_index(self, index: faiss.IndexFlatIP, lang: str):
        """
        Save the FAISS index to a file.
        """
        cache_key = self.get_cache_key()
        file_path = os.path.join(self.cache_dir, f'{cache_key}_{lang}_faiss_index.pkl')
        faiss.write_index(index, file_path)
        logging.info(f"Saved {lang} FAISS index to {file_path}")

    def load_faiss_index(self, lang: str) -> faiss.IndexFlatIP:
        """
        Load the FAISS index from a file.
        """
        cache_key = self.get_cache_key()
        file_path = os.path.join(self.cache_dir, f'{cache_key}_{lang}_faiss_index.pkl')
        if os.path.exists(file_path):
            index = faiss.read_index(file_path)
            logging.info(f"Loaded {lang} FAISS index from {file_path}")
            print(f"Loaded {lang} FAISS index from {file_path}")
            return index
        return None
    
    def load_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load the English and German datasets.
        """
        dataset_en = load_dataset(self.config.dataset_name, self.config.en_subset, streaming=True, trust_remote_code=True)
        dataset_de = load_dataset(self.config.dataset_name, self.config.de_subset, streaming=True, trust_remote_code=True)
        en_samples = self.sample_streaming_dataset(dataset_en['train'])
        de_samples = self.sample_streaming_dataset(dataset_de['train'])
        return en_samples, de_samples

    def sample_streaming_dataset(self, dataset, num_samples=None, buffer_size=None) -> List[Dict]:
        """
        Sample a subset of items from a streaming dataset.
        """
        num_samples = num_samples or self.config.num_samples
        buffer_size = buffer_size or self.config.buffer_size
        buffer = deque(maxlen=buffer_size)
        samples = []
        
        for item in dataset:
            if len(buffer) < buffer_size:
                buffer.append(item)
            else:
                idx = random.randint(0, buffer_size - 1)
                if idx < num_samples:
                    samples.append(item)
                    buffer[idx] = item
            
            if len(samples) >= num_samples:
                break
        
        return samples

    def preprocess_text(self, text: str, lang: str) -> List[str]:
        """
        Tokenize the text into sentences.
        """
        if lang == 'en':
            return self.english_tokenizer.tokenize(text)
        elif lang == 'de':
            return self.german_tokenizer.tokenize(text)
        else:
            raise ValueError("Unsupported language. Use 'en' for English or 'de' for German.")

    def filter_sentences(self, sentences: List[str]) -> List[str]:
        """
        Filter out sentences that are too short or too long.
        """
        sentences = [sent for sent in sentences if self.config.min_sentence_length <= len(sent.split()) <= self.config.max_sentence_length]
        # filter out the sentences with a "\n"
        sentences = [sent for sent in sentences if "\n" not in sent]
        return sentences


    def compute_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """
        Compute sentence embeddings using a transformer model.
        """
        return self.sentence_transformer.encode(sentences, convert_to_tensor=True)

    def prepare_faiss_index(self, embeddings: torch.Tensor) -> faiss.IndexFlatIP:
        """
        Prepare a FAISS index for the embeddings.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        faiss.normalize_L2(embeddings)
    
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        logging.info(f"Faiss index type: {type(index)}")

        index.add(embeddings)
        
        return index

    def lexical_diversity(self, sent1: str, sent2: str) -> float:
        """
        Compute the lexical diversity between two sentences.
        """
        words1 = [word.lower() for word in word_tokenize(sent1) 
                  if word.lower() not in self.stop_words and word not in string.punctuation]
        words2 = [word.lower() for word in word_tokenize(sent2) 
                  if word.lower() not in self.stop_words and word not in string.punctuation]
        
        intersection = set(words1).intersection(set(words2))
        union = set(words1).union(set(words2))
        return 1 - len(intersection) / len(union)


    def find_diverse_pairs_with_stats(self, sentences: List[str], embeddings: torch.Tensor, n_query: int = 1000, index: faiss.IndexFlatIP = None) -> Tuple[List[Tuple], Dict[str, Dict[str, float]]]:
        """
        Find diverse pairs of sentences based on their embeddings and lexical diversity.
        """

        if index is None:
            index = self.prepare_faiss_index(embeddings)
        
        # Sample a subset of embeddings for querying
        sample_size = min(n_query, len(embeddings))
        sample_indices = random.sample(range(len(embeddings)), sample_size)
        sample_embeddings = embeddings[sample_indices]
        sample_sentences = [sentences[i] for i in sample_indices]

        if isinstance(sample_embeddings, torch.Tensor):
            sample_embeddings = sample_embeddings.cpu().numpy()
        faiss.normalize_L2(sample_embeddings)

        similarities, indices = index.search(sample_embeddings, self.config.k)

        diverse_pairs = []
        all_lex_divs = []
        all_sem_dists = []

        for i, global_i in enumerate(sample_indices):
            max_lex_div = 0
            best_pair = None
            for j, sim in zip(indices[i], similarities[i]):
                if global_i != j and sim > self.config.semantic_threshold:
                    lex_div = self.lexical_diversity(sentences[global_i], sentences[j])
                    all_lex_divs.append(lex_div)
                    all_sem_dists.append(sim)
                    if lex_div > max_lex_div:
                        max_lex_div = lex_div
                        best_pair = (sentences[global_i], sentences[j], float(sim), lex_div)
            if best_pair:
                diverse_pairs.append(best_pair)

        dissimilarity_stats = {
            'lex_div': {
                'min': np.min(all_lex_divs) if all_lex_divs else None,
                'max': np.max(all_lex_divs) if all_lex_divs else None,
                'mean': np.mean(all_lex_divs) if all_lex_divs else None
            },
            'semantic_sim': {
                'min': np.min(all_sem_dists) if all_sem_dists else None,
                'max': np.max(all_sem_dists) if all_sem_dists else None,
                'mean': np.mean(all_sem_dists) if all_sem_dists else None
            }
        }

        return diverse_pairs, dissimilarity_stats

    def find_cross_lingual_pairs(self, en_embeddings: torch.Tensor, de_embeddings: torch.Tensor,
                             en_sentences: List[str], de_sentences: List[str], n_query: int = 1000) -> List[Tuple]:
        """
        Find cross-lingual pairs of sentences based on their embeddings,
        ensuring no sentence is used more than once.
        """
        all_embeddings = torch.cat([en_embeddings, de_embeddings])
        index = self.prepare_faiss_index(all_embeddings)

        # Sample a subset of embeddings for querying
        sample_size = min(n_query, len(all_embeddings))
        sample_indices = random.sample(range(len(all_embeddings)), sample_size)
        sample_embeddings = all_embeddings[sample_indices]

        if isinstance(sample_embeddings, torch.Tensor):
            sample_embeddings = sample_embeddings.cpu().numpy()
        faiss.normalize_L2(sample_embeddings)

        similarities, indices = index.search(sample_embeddings, self.config.k)

        pairs = []
        used_en_indices = set()
        used_de_indices = set()
        en_count = len(en_embeddings)

        for i, global_i in enumerate(sample_indices):
            for j, sim in zip(indices[i], similarities[i]):
                if sim <= self.config.cross_lingual_threshold:
                    continue

                if global_i < en_count and j >= en_count:
                    en_idx, de_idx = global_i, j - en_count
                elif global_i >= en_count and j < en_count:
                    en_idx, de_idx = j, global_i - en_count
                else:
                    continue

                if en_idx in used_en_indices or de_idx in used_de_indices:
                    continue

                pairs.append((en_sentences[en_idx], de_sentences[de_idx], float(sim)))
                used_en_indices.add(en_idx)
                used_de_indices.add(de_idx)

        return pairs

    def save_pairs_to_csv(self, pairs: List[Tuple], save_path: str, filename: str):
        """
        Save the sentence pairs to a CSV file.    
        """
        if not self.config.generate_csv:
            return

        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['Sentence 1', 'Sentence 2', 'Similarity Score', 'Lexical Diversity'])
            for pair in pairs:
                if len(pair) == 3:  # Cross-lingual pairs
                    writer.writerow([pair[0], pair[1], pair[2], 'N/A'])
                else:
                    writer.writerow(pair)

        logging.info(f"CSV file generated: {filepath}")

    def run(self, experiment_id: str) -> Tuple[List[Tuple], List[Tuple], List[Tuple], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Run the sentence pair finder pipeline.
        """
        en_embeddings, en_sentences = self.load_embeddings('en')
        de_embeddings, de_sentences = self.load_embeddings('de')

        if en_embeddings is None or de_embeddings is None:
            print("Loading datasets...")
            start_time = datetime.now()
            en_samples, de_samples = self.load_datasets()
            end_time = datetime.now()
            logging.info(f"Dataset loading completed in {(end_time - start_time).total_seconds()} seconds.")

            start_time = datetime.now()
            print("Preprocessing and filtering sentences...")
            en_sentences = [sent for doc in en_samples for sent in self.preprocess_text(doc['text'], 'en')]
            de_sentences = [sent for doc in de_samples for sent in self.preprocess_text(doc['text'], 'de')]
            
            en_sentences = self.filter_sentences(en_sentences)
            de_sentences = self.filter_sentences(de_sentences)
            
            if len(en_sentences) > self.config.max_sentences:
                en_sentences = random.sample(en_sentences, self.config.max_sentences)
            if len(de_sentences) > self.config.max_sentences:
                de_sentences = random.sample(de_sentences, self.config.max_sentences)
            
            print(f"Number of English sentences: {len(en_sentences)}")
            print(f"Number of German sentences: {len(de_sentences)}")

            end_time = datetime.now()
            logging.info(f"Preprocessing and filtering completed in {(end_time - start_time).total_seconds()} seconds.")

            start_time = datetime.now()
            print("Computing embeddings...")
            en_embeddings = self.compute_embeddings(en_sentences)
            de_embeddings = self.compute_embeddings(de_sentences)
            end_time = datetime.now()
            logging.info(f"Embedding computation completed in {(end_time - start_time).total_seconds()} seconds.")

            # Save the computed embeddings
            self.save_embeddings(en_embeddings, en_sentences, 'en')
            self.save_embeddings(de_embeddings, de_sentences, 'de')
        else:
            print("Loaded embeddings from cache.")

        start_time = datetime.now()
        print("Finding pairs...")
        n_query = self.config.num_query_samples

        # Try to load FAISS indexes
        en_index = self.load_faiss_index('en')
        de_index = self.load_faiss_index('de')

        if en_index is None:
            en_index = self.prepare_faiss_index(en_embeddings)
            self.save_faiss_index(en_index, 'en')
        if de_index is None:
            de_index = self.prepare_faiss_index(de_embeddings)
            self.save_faiss_index(de_index, 'de')

        en_diverse_pairs, en_stats = self.find_diverse_pairs_with_stats(en_sentences, en_embeddings, n_query, en_index)
        de_diverse_pairs, de_stats = self.find_diverse_pairs_with_stats(de_sentences, de_embeddings, n_query, de_index)
        cross_lingual_pairs = self.find_cross_lingual_pairs(en_embeddings, de_embeddings, en_sentences, de_sentences, n_query)
        end_time = datetime.now()
        logging.info(f"Pair finding completed in {(end_time - start_time).total_seconds()} seconds.")

        # Save pairs to CSV if enabled
        save_path = os.path.join(self.config.csv_output_dir, experiment_id)
        self.save_pairs_to_csv(en_diverse_pairs, save_path, 'en_diverse_pairs.csv')
        self.save_pairs_to_csv(de_diverse_pairs, save_path, 'de_diverse_pairs.csv')
        self.save_pairs_to_csv(cross_lingual_pairs, save_path, 'cross_lingual_pairs.csv')
        
        return en_diverse_pairs, de_diverse_pairs, cross_lingual_pairs, en_stats, de_stats

