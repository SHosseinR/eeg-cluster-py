import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from config import N_FEATURES_COMBINATION, N_FOLDS, N_TOP_FEATURES, RANDOM_STATE

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
        # print(f'{y_pred=}')
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_coefficients.append(clf.coef_[0])
    print(f'{accuracies=}')
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Return coefficients from last fold (representative)
    coefficients = all_coefficients[-1]
    
    return mean_accuracy, std_accuracy, coefficients


X = np.random.rand(40, 10)
X[:20, :] += np.random.rand(20, 10)
y = np.concatenate((np.zeros(20), np.ones(20)))
print(f'{X.shape=}, {y.shape=}')
evaluate_feature_triplet(X, y,  (0, 1, 3))