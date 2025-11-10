import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove quotes and special characters but keep Thai characters
    text = re.sub(r'[\"\"\'\'\#\@\$\&\*\(\)\[\]\{\}\<\>]', '', text)
    return text.strip()

def load_and_preprocess_data(train_file, test_file):
    """Load and preprocess the data"""
    # Load training data
    train_df = pd.read_csv(train_file, encoding='utf-8', sep=';', header=None, names=['review', 'rating'])
    train_df.columns = ['review', 'rating']
    
    # Load test data
    test_df = pd.read_csv(test_file, encoding='utf-8', sep=';')
    test_df.columns = ['reviewID', 'review']
    
    # Clean text
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    
    # Convert ratings to 0-based for XGBoost
    train_df['rating_label'] = train_df['rating'] - 1
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Rating distribution in training data:\n{train_df['rating'].value_counts().sort_index()}")
    
    return train_df, test_df

class BERTEmbeddingExtractor:
    def __init__(self, model_name='bert-base-multilingual-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
    
    def get_embeddings(self, texts, batch_size=8):
        """Extract BERT embeddings for a list of texts"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden states
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier and evaluate on validation set"""
    
    # Initialize and train XGBoost classifier
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softmax',
        num_class=5
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = xgb_model.predict(X_val)
    
    # Calculate F1 scores
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    f1_per_class = f1_score(y_val, y_pred, average=None)
    
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"F1 Score per class: {f1_per_class}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['1', '2', '3', '4', '5']))
    
    return xgb_model, f1_macro, f1_weighted, f1_per_class

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_location = "/Users/yeoboonhong/Documents/assignment/w_review_train.csv"
    test_location = "/Users/yeoboonhong/Documents/assignment/review_pred.csv"
    train_df, test_df = load_and_preprocess_data(train_location, test_location)
    
    # Extract BERT embeddings
    print("Extracting BERT embeddings...")
    bert_extractor = BERTEmbeddingExtractor()
    
    # Get embeddings for training data
    train_embeddings = bert_extractor.get_embeddings(train_df['cleaned_review'].tolist())
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_embeddings, 
        train_df['rating_label'].values,
        test_size=0.2,
        random_state=42,
        stratify=train_df['rating_label'].values
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # Train and evaluate model
    print("Training XGBoost classifier...")
    model, f1_macro, f1_weighted, f1_per_class = train_and_evaluate_model(X_train, y_train, X_val, y_val)
    
    # Predict on test data
    print("Predicting on test data...")
    test_embeddings = bert_extractor.get_embeddings(test_df['cleaned_review'].tolist())
    test_predictions = model.predict(test_embeddings)
    
    # Convert back to 1-5 scale
    test_df['predicted_rating'] = test_predictions + 1
    
    # Save predictions
    output_df = test_df[['reviewID', 'review', 'predicted_rating']]
    output_df.to_csv('predictions.csv', index=False, encoding='utf-8')
    
    print("\nPredictions saved to predictions.csv")
    print("\nSample predictions:")
    print(output_df.head(10))
    
    # Model performance summary
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Mean F1 Score (Macro): {f1_macro:.4f}")
    print(f"Mean F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Scores by class: {f1_per_class}")
    
    return model, bert_extractor, output_df

if __name__ == "__main__":
    model, bert_extractor, predictions = main()