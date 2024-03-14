#!/usr/bin/env python
# coding: utf-8

# file_name: optunaopt.py


import optuna
from tqdm.auto import tqdm
import time


def sklearn_opt(model, objective, X_train, y_train, n_trials, direction, 
                model_name='model', return_train_time=False, return_study=False):    
    '''
    Optimise a scikit-learn model with the Optuna library.

    The model is optimised based on a user-defined objective function.

    Parameters
    ----------
    model : class
        The scikit-learn model class (e.g., LogisticRegression, DecisionTree, etc.) that 
        you want to optimise.
    objective : function
        The objective function to be minimised or maximised during optimisation. This function should 
        take a single argument `trial` object, and it should return a numerical score to be optimised.
    X_train : array-like of shape (n_samples, n_features)
        The data to fit. For example, a list or an array.
    y_train : array-like of shape (n_samples,)
        The target variable to try to predict.
    n_trials : int
        The number of trials (iterations) for the optimisation process.
    direction : str, {'minimize', 'maximize'}
        Direction of optimisation. Set `minimize` for minimisation and `maximize` for maximisation.    
    model_name : str, default 'model'
        A string representing the name of the model.
    return_train_time : bool, default False
        If True, the function returns training time.        
    return_study : bool, default False
        If True, the function returns the optuna study.

    Returns
    -------
    tuple
        A tuple containing the optimised model and the training time.
        
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score

    >>> def objective(trial):
    >>>     # Define your objective function to be minimised or maximised        
    >>>     C = trial.suggest_float('C', 1e-3, 1e1, log=True)        
        
    >>>     clf = LogisticRegression(C=C)
    >>>     # Assume X_train, y_train are defined
    >>>     accs = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
        
    >>>     # Return a numerical score to be minimised or maximised
    >>>     return accs.mean()

    >>> # Assume X_train, y_train are defined
    >>> best_model = sklearn_clf_opt(LogisticRegression, objective, n_trials=50, 
    ...                              direction='maximize', X_train=X_train, y_train=y_train,
    ...                              model_name='MyLogisticRegression')
    '''
    # Set the optuna verbosity level to warning only
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create the optimisation study
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42),
                                direction=direction,
                                study_name=f'{model_name}_study')    

    # Optimise the model and show the progress bar
    with tqdm(total=n_trials, desc=f'Optimising {model_name}... ') as pbar:
        def callback(study, trial):
            pbar.update(1)    
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Print the best hyperparameter found
    print('Best hyperparameters:')
    display(study.best_params)    

    # Recreate the best model
    best_model = model(**study.best_params) 

    # Train the model and record the training time          
    train_start = time.time()
    best_model.fit(X_train, y_train)
    train_end = time.time()
    train_time = round((train_end - train_start) * 1000)

    if return_train_time:
        if return_study:
            return best_model, train_time, study
        else:
            return best_model, train_time
    else:
        if return_study:
            return best_model, study
        else:
            return best_model
