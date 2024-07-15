# train.py

import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import SimilarityBasedTripletDataset
from model import TripletLossModel
from transformers import AutoTokenizer
from config import MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SAVE_DIR, TRIPLET_DATASET_PATH, NUM_SAMPLES

def train():
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TripletLossModel(MODEL_NAME)
    print("Tokenizer and model loaded.")
    
    # Prepare dataset and dataloader
    try:
        dataset = SimilarityBasedTripletDataset.load_triplet_dataset(TRIPLET_DATASET_PATH, tokenizer)
        print(f"Loaded dataset from {TRIPLET_DATASET_PATH}")
    except FileNotFoundError:
        print(f"{TRIPLET_DATASET_PATH} not found. Creating new dataset.")
        SimilarityBasedTripletDataset.create_and_save_triplet_dataset(tokenizer, model.model, TRIPLET_DATASET_PATH, NUM_SAMPLES)
        dataset = SimilarityBasedTripletDataset.load_triplet_dataset(TRIPLET_DATASET_PATH, tokenizer)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Optimizer initialized.")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}...")
        model.train()
        total_loss = 0

        for batch in dataloader:
            anchor, positive, negative = batch
            anchor = {k: v.squeeze(0) for k, v in anchor.items()}
            positive = {k: v.squeeze(0) for k, v in positive.items()}
            negative = {k: v.squeeze(0) for k, v in negative.items()}

            optimizer.zero_grad()
            loss = model(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

    # Save the fine-tuned model
    print("Saving the fine-tuned model...")
    model.model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved in {SAVE_DIR}")

if __name__ == "__main__":
    train()
