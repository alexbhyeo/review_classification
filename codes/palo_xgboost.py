import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np

train_file_location = "/Users/yeoboonhong/Documents/assignment/w_review_train.csv"

# Load and preprocess the data

# Basic text cleaning function
def clean_text(text):
    if isinstance(text, str):
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove quotes if they surround the entire text
        text = text.strip('"')
    return text

# Parse the CSV data
def parse_data(filename):

    print(f"{'='*100}")
    reviews = pd.read_csv(filename, encoding='utf-8', sep=';', header=None, names=['review_text', 'rating'])
    print("length : ", len(reviews))
    print("header : ", reviews.dtypes)
    print(reviews.describe())

    print(f"{'='*100}")

    # Check for missing values
    print("Missing values:")
    print(reviews.isnull().sum())


    # Apply text cleaning
    reviews['cleaned_review'] = reviews['review_text'].apply(clean_text)

    return reviews

parsed_data = parse_data(train_file_location)


# Updated models configuration with proper label handling
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
}

def convert_labels_for_xgboost(y):
    """
    Convert labels from 1-5 to 0-4 for XGBoost
    """
    return y - 1

def convert_predictions_back(predictions):
    """
    Convert predictions back from 0-4 to 1-5
    """
    return predictions + 1

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Train and evaluate multiple machine learning models with proper label handling
    """
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Handle XGBoost label conversion
        if 'XGBoost' in name:
            y_train_adj = convert_labels_for_xgboost(y_train)
            y_test_adj = convert_labels_for_xgboost(y_test)
        else:
            y_train_adj = y_train
            y_test_adj = y_test
        
        # Train model
        model.fit(X_train, y_train_adj)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Convert predictions back for XGBoost
        if 'XGBoost' in name:
            y_pred = convert_predictions_back(y_pred)
            y_test_for_eval = y_test  # Use original labels for evaluation
        else:
            y_test_for_eval = y_test
        
        # Calculate F1 scores
        f1_macro = f1_score(y_test_for_eval, y_pred, average='macro')
        f1_weighted = f1_score(y_test_for_eval, y_pred, average='weighted')
        f1_per_class = f1_score(y_test_for_eval, y_pred, average=None)
        
        # Store results
        results[name] = {
            'model': model,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'predictions': y_pred,
            'needs_label_conversion': 'XGBoost' in name
        }
        
        print(f"{name}:")
        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        print(f"  Per-class F1: {dict(zip(range(1,6), f1_per_class))}")
        print("-" * 50)
    
    return results

# Advanced XGBoost with hyperparameter tuning
def create_optimized_xgboost():
    """
    Create optimized XGBoost model with better parameters for text classification
    """
    return xgb.XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=1,
        eval_metric='mlogloss',
        use_label_encoder=False,
        tree_method='hist'  # Faster training
    )

# Add optimized XGBoost to models
models['XGBoost Optimized'] = create_optimized_xgboost()



def main():
    # Assuming you have your data loaded in `df`
    df = parse_data(train_file_location)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    X_tfidf = tfidf.fit_transform(df['cleaned_review'])
    y = df['rating']  # This should be 1,2,3,4,5
    
    print(f"Unique labels in y: {sorted(y.unique())}")
    print(f"Label distribution:\n{y.value_counts().sort_index()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training models with proper label handling...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models)
    
    return results, tfidf

# Run the training
results, tfidf_vectorizer = main()


# Find the best model based on F1 score
def find_best_model(results):
    best_model_name = None
    best_f1 = 0
    
    for name, result in results.items():
        if result['f1_macro'] > best_f1:
            best_f1 = result['f1_macro']
            best_model_name = name
    
    return best_model_name, results[best_model_name]

best_model_name, best_model_result = find_best_model(results)
print(f"Best model: {best_model_name}")
print(f"Best macro F1 score: {best_model_result['f1_macro']:.4f}")

# Function to predict ratings for new data
def predict_ratings(new_data_file, best_model, tfidf_vectorizer, needs_label_conversion=False):
    """
    Predict ratings for new review data
    """
    # Load and preprocess the new data
    new_reviews = pd.read_csv(new_data_file, encoding='utf-8', sep=';', header=None, names=['reviewID', 'review_text'])
    
    print(f"Loaded {len(new_reviews)} reviews for prediction")
    print("\nSample of new reviews:")
    for i, (idx, row) in enumerate(new_reviews.head(3).iterrows()):
        print(f"ReviewID {row['reviewID']}: {row['review_text'][:100]}...")
    
    # Clean the text
    new_reviews['cleaned_review'] = new_reviews['review_text'].apply(clean_text)
    
    # Transform using the same TF-IDF vectorizer
    X_new_tfidf = tfidf_vectorizer.transform(new_reviews['cleaned_review'])
    
    # Make predictions
    predictions = best_model.predict(X_new_tfidf)
    
    # Convert predictions back if needed (for XGBoost)
    if needs_label_conversion:
        predictions = convert_predictions_back(predictions)
    
    # Add predictions to dataframe
    new_reviews['predicted_rating'] = predictions
    
    return new_reviews

# Predict ratings for the small_example.csv
print("\n" + "="*80)
print("PREDICTING RATINGS FOR small_example.csv")
print("="*80)


test_file_location = "/Users/yeoboonhong/Documents/assignment/review_pred.csv"

predicted_reviews = predict_ratings(
    test_file_location, 
    best_model_result['model'], 
    tfidf_vectorizer,
    needs_label_conversion=best_model_result['needs_label_conversion']
)

# Display results
print(f"\nPrediction Results:")
print(f"{'ReviewID':<10} {'Predicted Rating':<15} {'Review Excerpt':<50}")
print("-" * 80)

for _, row in predicted_reviews.iterrows():
    review_excerpt = row['review_text'][:47] + "..." if len(row['review_text']) > 50 else row['review_text']
    print(f"{row['reviewID']:<10} {row['predicted_rating']:<15} {review_excerpt:<50}")

# Show rating distribution
print(f"\nPredicted Rating Distribution:")
rating_counts = predicted_reviews['predicted_rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    print(f"Rating {rating}: {count} reviews")

# Save results to file
output_filename = "small_example_predictions.csv"
predicted_reviews[['reviewID', 'review_text', 'predicted_rating']].to_csv(output_filename, index=False, encoding='utf-8')
print(f"\nPredictions saved to: {output_filename}")

# Optional: Display detailed results for each review
print(f"\nDetailed Predictions:")
print("="*100)
for _, row in predicted_reviews.iterrows():
    print(f"\nReviewID: {row['reviewID']}")
    print(f"Predicted Rating: {row['predicted_rating']}")
    print(f"Review: {row['review_text'][:200]}...")
    print("-" * 100)


