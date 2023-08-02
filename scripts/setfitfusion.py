import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score,f1_score
from sentence_transformers import SentenceTransformer

def get_embedding_models(dms):
    """
    Returns a dictionary containing the SentenceTransformer models for each domain.
    
    Args:
    - dms (list): List of domain names
    
    Returns:
    - embedding_models (dict): Dictionary containing the SentenceTransformer models
    """
    embedding_models = {}

    for dm in dms:
        embedding_models[dm] = {}
        for dmy in dms:
            model_name = f"fathyshalab/reklambox-{dmy}-setfit"
            embedding_models[dm][model_name] = SentenceTransformer(model_name, use_auth_token=True)

    return embedding_models


def compute_weighted_embeddings(dms, test_ds, embedding_models,weights=None):
    """
    Computes the averaged embeddings for each sentence in the test dataset
    across all models.
    
    Args:
    - dms (list): List of domain names
    - test_ds (Dataset): Hugging Face Dataset containing the test set
    - embedding_models (dict): Dictionary containing the SentenceTransformer models
    
    Returns:
    - averaged_embeddings (dict): Dictionary containing the averaged embeddings for each sentence
    """
    # Initialize a dictionary to store the averaged embeddings
    averaged_embeddings = {}

    # Loop through each domain
    for dm in dms:
        # Initialize a dictionary to store the averaged embeddings for this domain
        averaged_embeddings[dm] = {}

        # Loop through each sentence in the test dataset
        for i in range(len(test_ds)):
            # Initialize a list to store the embeddings for this sentence across all models
            embeddings = []

            # Loop through each domain again
            for dmy in dms:
                # Get the embedding for this sentence from the current model
                embedding = embedding_models[dm][f"fathyshalab/reklambox-{dmy}-setfit"][i]
                embeddings.append(embedding)

            # Take the mean of the embeddings for this sentence across all models
            averaged_embedding = np.mean(embeddings, axis=0)

            # Add the averaged embedding to the dictionary
            averaged_embeddings[dm][i] = averaged_embedding
            
    return averaged_embeddings


def train_logistic_regression(dms, averaged_embeddings, test_ds):
    """
    Trains a logistic regression model on the averaged embeddings for each domain,
    and prints the classification report.
    
    Args:
    - dms (list): List of domain names
    - averaged_embeddings (dict): Dictionary containing the averaged embeddings for each sentence
    - test_ds (Dataset): Hugging Face Dataset containing the test set
    """
    # Loop through each domain
    metrics = {"accuracy":[],"balanced_accuracy":[],"f1":[]}
    for dm in dms:
        # Get the averaged embeddings and labels for this domain
        X = np.array(list(averaged_embeddings[dm].values()))
        y = test_ds["label"]

        # Train a logistic regression model on the averaged embeddings
        clf = LogisticRegression(random_state=42).fit(X, y)

        # Make predictions on the test set
        y_pred = clf.predict(X)

        # Print the classification report
        print(f"Domain: {dm}")
        print(classification_report(y, y_pred))
        metrics["f1"].append(f1_score(y,y_pred))
        metrics["accuracy"].append(accuracy_score(y,y_pred))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y,y_pred))
    return metrics



 
