import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
from config import MODEL_NAME, MAX_LENGTH, THRESHOLD, NUM_SAMPLES, TRIPLET_DATASET_PATH

class SimilarityBasedTripletDataset(Dataset):
    def __init__(self, sentences, similarity_matrix, tokenizer, max_length=MAX_LENGTH, threshold=THRESHOLD):
        self.sentences = sentences
        self.similarity_matrix = similarity_matrix
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.threshold = threshold

    @staticmethod
    def _get_embeddings(sentences, tokenizer, model):
        print(f"Computing embeddings for {len(sentences)} sentences...")
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        print("Embeddings computed.")
        return embeddings

    @staticmethod
    def create_and_save_triplet_dataset(tokenizer, model, save_path, num_samples):
        print(f"Creating dataset with {num_samples} samples from streaming data...")
        sentences = []
        dataset = load_dataset('joelniklaus/Multi_Legal_Pile', "en_caselaw", split='train', streaming=True)
        
        # Collect a specified number of samples from the streaming dataset
        for i, example in enumerate(dataset):
            sentences.append(example['text'])
            if len(sentences) % 100 == 0:
                print(f"Collected {len(sentences)} samples so far...")
            if len(sentences) >= num_samples:
                break

        print(f"Collected {len(sentences)} samples. Computing embeddings in batches...")
        
        # Compute embeddings in batches to handle memory constraints
        batch_size = 100
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = SimilarityBasedTripletDataset._get_embeddings(batch, tokenizer, model)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)

        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        print("Similarity matrix computed.")

        data = {
            'sentences': sentences,
            'similarity_matrix': similarity_matrix
        }
        torch.save(data, save_path)
        print(f"Dataset saved at {save_path}")

    @staticmethod
    def load_triplet_dataset(load_path, tokenizer):
        print(f"Loading dataset from {load_path}...")
        data = torch.load(load_path)
        print("Dataset loaded.")
        return SimilarityBasedTripletDataset(
            sentences=data['sentences'],
            similarity_matrix=data['similarity_matrix'],
            tokenizer=tokenizer
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        anchor = self.sentences[idx]
        similarities = self.similarity_matrix[idx]

        positive_indices = [i for i, sim in enumerate(similarities) if sim > self.threshold and i != idx]
        negative_indices = [i for i, sim in enumerate(similarities) if sim < self.threshold]

        if not positive_indices:
            positive_indices = [i for i in range(len(self.sentences)) if i != idx]

        positive = self.sentences[random.choice(positive_indices)]
        negative = self.sentences[random.choice(negative_indices)]

        anchor_enc = self.tokenizer(anchor, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        positive_enc = self.tokenizer(positive, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        negative_enc = self.tokenizer(negative, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return anchor_enc, positive_enc, negative_enc
