from autogluon.tabular import TabularDataset, TabularPredictor
from skfp.fingerprints import ECFPFingerprint,  TopologicalTorsionFingerprint, MordredFingerprint, ERGFingerprint
from skfp.preprocessing import MolFromSmilesTransformer
import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    average_precision_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def clean_features(train_df, test_df, feature_columns):
    train = train_df.copy()
    test = test_df.copy()

    for col in feature_columns:
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    return train, test


def train_model(train_data, test_data, eval_metric, problem_type, task_name, experiment_type):
    train_smiles, train_y = train_data['smiles_standarized'], train_data['label']
    test_smiles, test_y = test_data['smiles_standarized'], test_data['label']

    mol_from_smiles = MolFromSmilesTransformer()
    mols_train = mol_from_smiles.transform(train_smiles)
    mols_test  = mol_from_smiles.transform(test_smiles)

    fp_classes = [
        ECFPFingerprint,
        ERGFingerprint,
        TopologicalTorsionFingerprint,
        MordredFingerprint
    ]
    fp_names = [
        "ECFP4",
        "ERG",
        "TopoTorsion",
        "Mordred"
    ]

    fp_train_list = []
    fp_test_list = []

    for i, (fp_cls, name) in enumerate(zip(fp_classes, fp_names)):
        fp = fp_cls(n_jobs=-1)
        train_fp = pd.DataFrame(fp.transform(mols_train))
        test_fp = pd.DataFrame(fp.transform(mols_test))

        train_fp.columns = [f"{name}_{c}" for c in train_fp.columns]
        test_fp.columns = [f"{name}_{c}" for c in test_fp.columns]

        feature_cols = train_fp.columns.tolist()
        train_fp_clean, test_fp_clean = clean_features(
            train_fp, test_fp, feature_cols
        )

        fp_train_list.append(train_fp_clean)
        fp_test_list.append(test_fp_clean)

    X_train = pd.concat(fp_train_list, axis=1)
    X_test = pd.concat(fp_test_list, axis=1)

    train_ag = X_train.copy()
    test_ag = X_test.copy()
    train_ag['label'] = train_y.values
    test_ag['label'] = test_y.values

    train_ag = TabularDataset(train_ag)
    test_ag = TabularDataset(test_ag)

    save_path = os.path.join(f"../model/model-{experiment_type}", f"{task_name}_{'_'.join(fp_names)}")
    os.makedirs(save_path, exist_ok=True)
    print(save_path)

    predictor = TabularPredictor(
        label='label',
        eval_metric=eval_metric,
        problem_type=problem_type,
        path=save_path,
    )

    predictor.fit(
        train_data=train_ag,
        presets='best_quality',
        num_cpus=64
        # presets='medium_quality',
        # num_bag_folds=8
    )

    
    best_model = predictor.get_model_best()
    
    best_val_score = predictor.leaderboard().set_index('model').loc[best_model, 'score_val']

    y_pred = predictor.predict(test_ag, model=best_model)

    
    pred_df = pd.DataFrame({
        'smiles_standarized': test_smiles.values,
        'y_true': test_y.values,
        'y_pred': y_pred.values
    })

    
    if problem_type == "binary":
        y_pred_proba = predictor.predict_proba(test_ag, model=best_model)
        y_pred_prob = y_pred_proba.iloc[:, 1]
        pred_df['y_pred_prob'] = y_pred_prob.values
        metrics = for_classification(task_name, best_model, problem_type, test_y, y_pred, y_pred_prob, best_val_score)
    else:
        metrics = for_regression(task_name, best_model, problem_type, test_y, y_pred, best_val_score)

    
    pred_save_dir = f"../metrics/metrics_{experiment_type}"
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_save_path = os.path.join(pred_save_dir, f"{task_name}_predictions_{experiment_type}.csv")
    pred_df.to_csv(pred_save_path, index=False)
    print(f"Predictions saved to: {pred_save_path}")
    # ======================================================

    return metrics


def for_classification(task_name, model_name, problem_type, y_test, y_pred, y_pred_prob, best_score):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2
    f2 = (5 * precision_score(y_test, y_pred, zero_division=0) * sensitivity) / \
         (4 * precision_score(y_test, y_pred, zero_division=0) + sensitivity) if (precision_score(y_test, y_pred, zero_division=0) + sensitivity) > 0 else 0.0

    return {
        "Task": task_name, "model": model_name, "Learning Task": problem_type,
        "score_val": best_score,
        "ROC AUC": roc_auc_score(y_test, y_pred_prob),
        "AUPRC": average_precision_score(y_test, y_pred_prob),
        "Balanced Accuracy": balanced_acc,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F2 Score": f2,
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Kappa": cohen_kappa_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Log Loss": log_loss(y_test, y_pred_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

def for_regression(task_name, model_name, problem_type, y_test, y_pred, best_score):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr = pearsonr(y_test, y_pred)[0]
    spearman_corr = spearmanr(y_test, y_pred)[0]
    rae = mae / np.mean(np.abs(y_test - np.mean(y_test)))
    rse = mse / np.mean(np.square(y_test - np.mean(y_test)))
    nrmse = rmse / (np.max(y_test) - np.min(y_test)) if np.max(y_test) != np.min(y_test) else 0.0
    explained_var = explained_variance_score(y_test, y_pred)

    return {
        "Task": task_name, "model": model_name, "Learning Task": problem_type,
        "score_val": best_score,
        "RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2,
        "Explained Variance": explained_var,
        "Pearson Corr": pearson_corr, "Spearman Corr": spearman_corr,
        "RAE": rae, "RSE": rse,
    }

# =============================================================================
# main
# =============================================================================
def main():
    TASK_NAME = sys.argv[1]    # task_name
    experiment_type = "best"    # experiment type (e.g. only model / bagging / stacking / all)
    TRAIN_CSV = f"../scaffold_split_results/{TASK_NAME}_train.csv"
    TEST_CSV = f"../scaffold_split_results/{TASK_NAME}_test.csv"

    train_df = TabularDataset(TRAIN_CSV)
    test_df = TabularDataset(TEST_CSV)

    unique_labels = train_df['label'].nunique()
    if unique_labels == 2:
        problem_type = "binary"
        eval_metric = "roc_auc"
        print("classification detected!")
    else:
        problem_type = "regression"
        eval_metric = "root_mean_squared_error"
        print("regression detected!")

    metrics = train_model(
        train_data=train_df, test_data=test_df,
        eval_metric=eval_metric, problem_type=problem_type, task_name=TASK_NAME, experiment_type=experiment_type
    )

    result_df = pd.DataFrame([metrics])
    result_df.to_csv(f"../metrics/metrics_{experiment_type}/{TASK_NAME}_results_{experiment_type}.csv", index=False)
    print(f"../metrics/metrics_{experiment_type}/{TASK_NAME}_results_{experiment_type}.csv")
    print("training completed!")

if __name__ == "__main__":
    main()
