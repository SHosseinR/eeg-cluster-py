"""
Classification and feature selection utilities
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from config import N_FEATURES_COMBINATION, N_FOLDS, N_TOP_FEATURES, RANDOM_STATE
from tqdm import tqdm

def evaluate_feature_triplet(X, y, feature_indices, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """
    Evaluate a triplet of features using cross-validation.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_indices : tuple
        Indices of the 3 features to use
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns
    -------
    mean_accuracy : float
        Average accuracy across folds
    std_accuracy : float
        Standard deviation of accuracy
    coefficients : ndarray
        Logistic regression coefficients (from last fold)
    """
    # Select features
    X_subset = X[:, feature_indices]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accuracies = []
    all_coefficients = []
    
    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_coefficients.append(clf.coef_[0])
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Return coefficients from last fold (representative)
    coefficients = all_coefficients[-1]
    
    return mean_accuracy, std_accuracy, coefficients


def find_best_feature_triplets(X, y, feature_names, n_top=N_TOP_FEATURES, verbose=True):
    """
    Find the best triplets of features for classification.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_names : list
        List of feature names
    n_top : int
        Number of top triplets to return
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with top feature triplets and their accuracies
    all_results : list
        List of all results (for further analysis)
    """
    n_features = X.shape[1]
    # print(f'{n_features=}')
    # Generate all combinations of 3 features
    all_triplets = list(combinations(range(n_features), N_FEATURES_COMBINATION))
    n_triplets = len(all_triplets)
    
    if verbose:
        print(f"\nEvaluating {n_triplets} feature triplets...")
        print(f"Using {N_FOLDS}-fold cross-validation")
        print("=" * 80)
    
    results = []
    
    for i, triplet in tqdm(enumerate(all_triplets)):
        if verbose and (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{n_triplets} ({(i+1)/n_triplets*100:.1f}%)")
        
        # Evaluate triplet
        mean_acc, std_acc, coeffs = evaluate_feature_triplet(X, y, triplet)
        
        # Store results
        triplet_names = [feature_names[idx] for idx in triplet]
        results.append({
            'triplet_indices': triplet,
            'triplet_names': triplet_names,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'coefficients': coeffs
        })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    # Create DataFrame with top results
    top_results = results[:n_top]
    
    df_data = []
    for rank, result in enumerate(top_results, 1):
        df_data.append({
            'Rank': rank,
            'Features': ' + '.join(result['triplet_names']),
            'Accuracy': f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}"
        })
    
    results_df = pd.DataFrame(df_data)
    
    if verbose:
        print("\n" + "=" * 80)
        print("TOP 10 FEATURE TRIPLETS:")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
    
    return results_df, results


def get_best_triplet_details(results_list, rank=1):
    """
    Get detailed information about a specific ranked triplet.
    
    Parameters
    ----------
    results_list : list
        Output from find_best_feature_triplets (all_results)
    rank : int
        Rank of the triplet to retrieve (1-indexed)
        
    Returns
    -------
    triplet_info : dict
        Dictionary with triplet details
    """
    if rank < 1 or rank > len(results_list):
        raise ValueError(f"Rank must be between 1 and {len(results_list)}")
    
    result = results_list[rank - 1]
    
    return {
        'rank': rank,
        'feature_names': result['triplet_names'],
        'feature_indices': result['triplet_indices'],
        'accuracy': result['mean_accuracy'],
        'accuracy_std': result['std_accuracy'],
        'coefficients': result['coefficients']
    }


def perform_final_classification(X, y, feature_indices, random_state=RANDOM_STATE):
    """
    Perform final classification with best features on entire dataset.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_indices : tuple or list
        Indices of features to use
    random_state : int
        Random seed
        
    Returns
    -------
    model : LogisticRegression
        Trained model
    scaler : StandardScaler
        Fitted scaler
    train_accuracy : float
        Training accuracy
    """
    # Select features
    X_subset = X[:, feature_indices]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Train model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Compute training accuracy
    y_pred = model.predict(X_scaled)
    train_accuracy = accuracy_score(y, y_pred)
    
    return model, scaler, train_accuracy


def analyze_feature_importance(coefficients, feature_names):
    """
    Analyze and rank feature importance from logistic regression.
    
    Parameters
    ----------
    coefficients : ndarray
        Logistic regression coefficients
    feature_names : list
        Names of features
        
    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with feature importance analysis
    """
    abs_coeffs = np.abs(coefficients)
    sorted_indices = np.argsort(abs_coeffs)[::-1]
    
    df_data = []
    for idx in sorted_indices:
        df_data.append({
            'Feature': feature_names[idx],
            'Coefficient': coefficients[idx],
            'Abs_Coefficient': abs_coeffs[idx],
            'Importance_Rank': len(df_data) + 1
        })
    
    importance_df = pd.DataFrame(df_data)
    
    return importance_df


def create_classification_report(X, y, feature_names, results_list, output_path=None):
    """
    Create a comprehensive classification report.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    feature_names : list
        Feature names
    results_list : list
        All results from feature selection
    output_path : str, optional
        Path to save report
        
    Returns
    -------
    report_dict : dict
        Dictionary with report information
    """
    # Get best triplet
    best = get_best_triplet_details(results_list, rank=1)
    
    # Train final model
    model, scaler, train_acc = perform_final_classification(
        X, y, best['feature_indices']
    )
    
    # Analyze importance
    importance_df = analyze_feature_importance(
        best['coefficients'], best['feature_names']
    )
    
    report_dict = {
        'best_triplet': best,
        'importance_df': importance_df,
        'train_accuracy': train_acc,
        'model': model,
        'scaler': scaler
    }
    
    if output_path:
        # Save report as text
        with open(output_path, 'w') as f:
            f.write("CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Best Feature Triplet (Rank 1):\n")
            f.write(f"  Features: {', '.join(best['feature_names'])}\n")
            f.write(f"  Cross-validation Accuracy: {best['accuracy']:.4f} ± {best['accuracy_std']:.4f}\n")
            f.write(f"  Training Accuracy: {train_acc:.4f}\n\n")
            
            f.write("Feature Importance:\n")
            f.write(importance_df.to_string(index=False))
            f.write("\n\n")
            
        print(f"Saved classification report: {output_path}")
    
    return report_dict
