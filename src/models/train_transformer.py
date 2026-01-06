"""
Transformer Models for Clarity Score Prediction

Uses XLM-RoBERTa for multilingual clarity prediction across English, Spanish, and Russian.

Usage:
    python src/models/train_transformer.py
    python src/models/train_transformer.py --model xlm-roberta-base --epochs 5
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse


class ClarityDataset(Dataset):
    """Dataset for clarity prediction with transformer models."""
    
    def __init__(self, texts, clarity_scores, tokenizer, max_length=256):
        """
        Args:
            texts: List of instruction texts
            clarity_scores: List of clarity scores
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.clarity_scores = clarity_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.clarity_scores[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class TransformerClarityModel(nn.Module):
    """Transformer-based model for clarity prediction."""
    
    def __init__(self, model_name='xlm-roberta-base', dropout=0.1):
        """
        Args:
            model_name: Name of the pre-trained transformer model
            dropout: Dropout rate for the regression head
        """
        super(TransformerClarityModel, self).__init__()
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from transformer config
        hidden_size = self.transformer.config.hidden_size
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            predictions: Clarity scores [batch_size, 1]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Predict clarity score
        predictions = self.regressor(pooled_output)
        
        return predictions.squeeze(-1)


class TransformerTrainer:
    """Train and evaluate transformer models for clarity prediction."""
    
    def __init__(
        self,
        model_name='xlm-roberta-base',
        project_root='.',
        batch_size=8,
        max_length=256,
        learning_rate=2e-5,
        num_epochs=2,
        warmup_steps=500,
        device=None
    ):
        """
        Args:
            model_name: Name of the transformer model
            project_root: Root directory of the project
            batch_size: Training batch size
            max_length: Maximum sequence length
            learning_rate: Learning rate for AdamW optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.model_name = model_name
        self.root = Path(project_root)
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        
        # Paths
        self.splits_path = self.root / "data" / "splits"
        self.models_path = self.root / "models" / "transformer"
        self.results_path = self.root / "artifacts" / "results"
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Will be initialized in load_data
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def load_data(self):
        """Load train/val/test splits and create data loaders."""
        print("Loading data...")
        train_df = pd.read_csv(self.splits_path / "train.csv")
        val_df = pd.read_csv(self.splits_path / "val.csv")
        test_df = pd.read_csv(self.splits_path / "test.csv")
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        # Initialize tokenizer
        print(f"\nLoading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = ClarityDataset(
            train_df['text'].tolist(),
            train_df['clarity_score'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = ClarityDataset(
            val_df['text'].tolist(),
            val_df['clarity_score'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = ClarityDataset(
            test_df['text'].tolist(),
            test_df['clarity_score'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"  Created {len(self.train_loader)} training batches")
        print(f"  Created {len(self.val_loader)} validation batches")
        print(f"  Created {len(self.test_loader)} test batches")
    
    def initialize_model(self):
        """Initialize the transformer model."""
        print(f"\nInitializing model: {self.model_name}")
        self.model = TransformerClarityModel(self.model_name).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def train(self):
        """Train the model."""
        print("\n" + "="*60)
        print("Training Transformer Model")
        print("="*60)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_mae = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            with tqdm(self.train_loader, desc="Training") as pbar:
                for batch in pbar:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    predictions = self.model(input_ids, attention_mask)
                    loss = criterion(predictions, labels)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Track loss
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader)
            
            # Print metrics
            print(f"\nTrain Loss: {avg_train_loss:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Val MAE:    {val_metrics['mae']:.4f}")
            print(f"Val R²:     {val_metrics['r2']:.4f}")
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_r2'].append(val_metrics['r2'])
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                self.save_model('best_model')
                print(f"✓ Saved best model (MAE: {best_val_mae:.4f})")
        
        # Save final model
        self.save_model('final_model')
        
        return history
    
    def evaluate(self, data_loader):
        """Evaluate the model on a data loader."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        mae = mean_absolute_error(all_labels, all_predictions)
        r2 = r2_score(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'r2': r2,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def test(self):
        """Evaluate on test set."""
        print("\n" + "="*60)
        print("Testing on Test Set")
        print("="*60)
        
        test_metrics = self.evaluate(self.test_loader)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        
        return test_metrics
    
    def save_model(self, name):
        """Save model checkpoint."""
        checkpoint_path = self.models_path / f"{name}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length
        }, checkpoint_path)
    
    def save_results(self, history, test_metrics):
        """Save training results to JSON."""
        results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps
            },
            'history': history,
            'test_metrics': {
                'loss': test_metrics['loss'],
                'mae': test_metrics['mae'],
                'r2': test_metrics['r2']
            }
        }
        
        results_path = self.results_path / f"transformer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train transformer model for clarity prediction')
    parser.add_argument('--model', type=str, default='xlm-roberta-base',
                        help='Transformer model name (default: xlm-roberta-base)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length (default: 256)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TransformerTrainer(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    # Load data and initialize model
    trainer.load_data()
    trainer.initialize_model()
    
    # Train
    history = trainer.train()
    
    # Test
    test_metrics = trainer.test()
    
    # Save results
    trainer.save_results(history, test_metrics)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {trainer.models_path}")
    print(f"Results saved to: {trainer.results_path}")


if __name__ == "__main__":
    main()