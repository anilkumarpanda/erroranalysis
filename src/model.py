# Code to tune the models
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from probatus.feature_elimination import EarlyStoppingShapRFECV
from yellowbrick.classifier import DiscriminationThreshold
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_monotone_constraints(data_dict,target, corr_threshold=0.1):
    """
    Method to get monotone constraints.
    
    Args:
        data_dict(dict) : Dictionary containing the training and testing data.
        target(str) : Target variable.
        corr_threshold(float) : Correlation threshold.
        Returns:
            monotone_constraints(dict) : Dictionary containing the monotone constraints.
    """
    data = data_dict["xtrain"].copy()
    data[target] = data_dict["ytrain"]

    corr = pd.Series(data.corr(method='spearman')[target]).drop(target)
    monotone_constraints = tuple(np.where(corr < -corr_threshold, -1,
                                          np.where(corr > corr_threshold, 1, 0)))
    return monotone_constraints


def select_features(data, n_features=16):
    """
    Method for feature selection.
    Args:
        data(dict) : Dictionary containing the training and testing data.
        n_features (int): Number of features to return.
    """
    # Simple feature selection strategy to ensure that the features used in the model are good.

    clf = XGBClassifier(max_depth=3,use_label_encoder=False,objective="binary:logistic")
    fs_param_grid = {
        "n_estimators": [5, 7, 10],
        "num_leaves": [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(clf, fs_param_grid)

    # Run feature elimination
    shap_elimination = EarlyStoppingShapRFECV(
        clf=search,
        step=0.2,
        cv=3,
        scoring="roc_auc",
        early_stopping_rounds=5,
        n_jobs=4,
    )

    shap_elimination.fit_compute(data["xtrain"], data["ytrain"])
    selected_feats = shap_elimination.get_reduced_features_set(num_features=n_features)
    print(f"Selected features : {selected_feats} ")
    return selected_feats, shap_elimination.plot()


def tune_parameters(data, model):
    """
    Method to tune the hyperparameters.

    Args:
        data(dict) : Dictionary containing the training and testing data.
        model(xgb.XGBClassifier): XGBClassifier.
    """
    # We will use a intial fixed set of parameters, these can be parameterised as well.
    # Better still use Optune here.

    parameters = {
        "eta": [0.01, 0.015, 0.025, 0.05, 0.1],
        "gamma": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "max_depth": [2, 3, 5, 7, 9, 12, 15, 17, 25],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "lambda": [0.01, 0.1, 1.0],
        "alpha": [0, 0.1, 0.5, 1.0],
    }

    # Instantiate the gridsearch
    hgb_grid = RandomizedSearchCV(
        model,
        parameters,
        n_jobs=5,
        random_state=345,
        cv=3,
        scoring="roc_auc",
        verbose=2,
    )
    hgb_grid.fit(data["xtrain"], data["ytrain"])
    # Return the best parameters and train the model.
    return hgb_grid.best_params_


def show_model_results(data, model,calc_threshold=False):
    """
    Show the model results.
    Args :
        data(dict) : Dictionary containing the training and testing data.
        model(xgboost.XGBClassifier) : Model used to train and evaluate data.
        metrics(dict): Metrics used to evaluate the model.Default is roc_auc.
    Returns :
        model(xgboost.XGBClassifier) : Return the trained model.
    """
    # Todo : Add cross validation.

    model.fit(data["xtrain"], data["ytrain"])

    y_train_proba = model.predict_proba(data["xtrain"])[:, 1]
    y_test_proba = model.predict_proba(data["xtest"])[:, 1]
    y_test_pred = model.predict(data["xtest"])

    print(f"Train ROC-AUC score : {roc_auc_score(data['ytrain'],y_train_proba)}")
    print(f"Test ROC-AUC score : {roc_auc_score(data['ytest'],y_test_proba)}")
    print(f"Test Accuracy socre : {accuracy_score(data['ytest'],y_test_pred)}")
    # Add other metrics as required.

    # The outputs are probabilites, however we would like to work with predictions.
    # Hence, lets convert the probas to predictions.
    if calc_threshold:
        visualizer = DiscriminationThreshold(model, quantiles=np.array([0.25, 0.5, 0.75]),exclude=['queue_rate'])
        visualizer.fit(data["xtrain"], data["ytrain"])
        visualizer.show()

    return model
