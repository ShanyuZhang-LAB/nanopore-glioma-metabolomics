#!/usr/bin/env python3
"""Reproduce the original nanopore glioma machine-learning analysis.

The analytical settings are preserved from the supplied original script.
Run this file from any working directory. Inputs are in folder 1 and results in folder 3.
The supplied pretrained model in folder 4 is never modified.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['Relative_Blockade', 'Current_SD', 'Current_Range']
SPLIT_SEED = 520
CV_SEED = 42
CV_FOLDS = 10
LEARNING_CURVE_FOLDS = 5
LEARNING_CURVE_SIZES = np.linspace(0.1, 1.0, 10)
# Preserve the original training class labels used in the archived model.
CLASS_LABELS = {'D-2-HG': 'D-2-HG.xlsx', 'Others': '\u975e\u7a81\u53d8\u4e09\u7ec4N.xlsx'}


def find_project_root():
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / '1_Training_Feature_Matrices').is_dir():
            return parent
    raise FileNotFoundError('Cannot locate 1_Training_Feature_Matrices above this script.')


def read_matrix(path):
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, encoding='utf-8-sig')
    return pd.read_excel(path)


def load_training_data(folder):
    # Explicit file order preserves the intended original D-2-HG / background ordering.
    files = []
    for stem in ('D-2-HG', 'Others'):
        matches = [p for p in folder.iterdir() if p.is_file()
                   and p.stem.casefold() == stem.casefold()
                   and p.suffix.lower() in ('.xlsx', '.xls', '.csv')]
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected exactly one {stem}.xlsx/.xls/.csv in {folder}; found {len(matches)}.')
        files.append((stem, matches[0]))
    frames = []
    for stem, path in files:
        df = read_matrix(path)
        if 'IsNoise' in df.columns:
            df = df.loc[pd.to_numeric(df['IsNoise'], errors='coerce') != -1].copy()
        missing = [col for col in FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f'{path.name} is missing required features: {missing}')
        X = df[FEATURES].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(f'{path.name} contains non-finite feature values.')
        X['source_file'] = CLASS_LABELS[stem]
        frames.append(X)
        print(f'Loaded {path.name}: {len(X)} events')
    combined = pd.concat(frames, ignore_index=True)
    return combined[FEATURES], combined['source_file']


def candidate_models():
    # Unchanged estimators and hyperparameters from the original analysis.
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
    }


def save_confusion_matrix(y_true, y_pred, title, output_folder, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', vmin=0, vmax=1, ax=ax,
                xticklabels=['D-2-HG', 'Others'], yticklabels=['D-2-HG', 'Others'],
                annot_kws={'size': 12})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted class', fontsize=12)
    ax.set_ylabel('True class', fontsize=12)
    fig.tight_layout()
    path = output_folder / ('confusion_matrix_' + re.sub(r'[^A-Za-z0-9_-]+', '_', title) + '.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(cm, index=['D-2-HG', 'Others'], columns=['D-2-HG', 'Others']).to_csv(path.with_suffix('.csv'))
    return path


def save_learning_curve(model, X, y, output_folder):
    # Original implementation: ALL training matrices, 5-fold integer CV,
    # ten fractions from 0.1 to 1.0, accuracy scoring, n_jobs=-1.
    sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=LEARNING_CURVE_FOLDS,
        train_sizes=LEARNING_CURVE_SIZES, scoring='accuracy', n_jobs=-1)
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    ax.fill_between(sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(sizes, test_mean, color='green', marker='s', markersize=5, label='Validation accuracy')
    ax.fill_between(sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    ax.set_title('Random Forest')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.savefig(output_folder / 'learning_curve_Random_Forest.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'train_sizes': sizes, 'training_accuracy': train_mean, 'train_std': train_std,
                  'validation_accuracy': test_mean, 'test_std': test_std}).to_csv(
        output_folder / 'learning_curve_data_Random_Forest.csv', index=False)


def save_feature_importance(model, output_folder):
    # Original impurity-based feature importance from the fitted 80% model.
    importance = np.asarray(model.feature_importances_)
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Random Forest feature importance')
    ax.bar(range(len(FEATURES)), importance[order])
    ax.set_xticks(range(len(FEATURES)), np.asarray(FEATURES)[order], rotation=90)
    fig.tight_layout()
    fig.savefig(output_folder / 'feature_importance_Random_Forest.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'feature_names': np.asarray(FEATURES)[order], 'importances': importance[order]}).to_csv(
        output_folder / 'feature_importance_data_Random_Forest.csv', index=False)


def main():
    root = find_project_root()
    output = root / '3_Machine_Learning_Training_results'
    output.mkdir(parents=True, exist_ok=True)
    X, y = load_training_data(root / '1_Training_Feature_Matrices')
    # Original global 80/20 split and 10-fold stratified CV on training only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y)
    nfold = min(CV_FOLDS, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=CV_SEED)
    rows = []
    models = candidate_models()
    for name, estimator in models.items():
        accuracies = []
        for train_idx, test_idx in cv.split(X_train, y_train):
            fitted = clone(estimator).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            predicted = fitted.predict(X_train.iloc[test_idx])
            accuracies.append(accuracy_score(y_train.iloc[test_idx], predicted))
        rows.append({'Model': name, 'CV_Mean_Accuracy': float(np.mean(accuracies)),
                     'CV_SD': float(np.std(accuracies)),
                     'Fold_Accuracies': json.dumps([float(x) for x in accuracies])})
        print(f'{name}: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}')
    pd.DataFrame(rows).to_csv(output / 'model_training_accuracy.csv', index=False)
    best = max(rows, key=lambda row: row['CV_Mean_Accuracy'])
    print(f'Best model: {best["Model"]} (CV accuracy: {best["CV_Mean_Accuracy"]:.4f})')
    if best['Model'] != 'Random Forest':
        raise RuntimeError('Random Forest was not selected. Check input data/order and environment before reproducing the paper figures.')
    fitted = clone(models['Random Forest']).fit(X_train, y_train)
    labels = [CLASS_LABELS['D-2-HG'], CLASS_LABELS['Others']]
    # Save only the held-out test-set confusion matrix for the best model.
    save_confusion_matrix(y_test, fitted.predict(X_test),
                          'Random Forest holdout set', output, labels)
    print(f'Random Forest holdout accuracy: {accuracy_score(y_test, fitted.predict(X_test)):.4f}')
    save_learning_curve(fitted, X, y, output)
    save_feature_importance(fitted, output)
    print(f'All outputs saved to: {output}')
    print('The supplied 4_Trained_Models.joblib has not been modified.')


if __name__ == '__main__':
    main()
