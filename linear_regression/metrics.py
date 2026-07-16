import numpy as np

def _validate_inputs(y_true : np.ndarray, y_pred :np.ndarray):
    ''' Validates Input by Comparing Shapes '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            'y_true and y_pred must have same shape'
        )
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError(
                'Inputs must not contain NaN values'
                )
    return y_true, y_pred
    
def mean_absolute_error(y_true : np.ndarray, y_pred :np.ndarray) -> float:
    ''' Returns Mean Absolute Error = (1/n) Σ (|y_true(i) - y_pred(i)|)'''
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred)).item()

def mean_squared_error(y_true : np.ndarray, y_pred :np.ndarray) -> float:
    ''' Returns Mean Squared Error = (1/n) Σ (y_true(i) - y_pred(i))^2'''
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.square(y_true - y_pred)).item()

def root_mean_squared_error(y_true : np.ndarray, y_pred :np.ndarray) -> float:
    ''' Returns Root Mean Squared Error = √((1/n) Σ (y_true(i) - y_pred(i))^2)'''
    return np.sqrt(mean_squared_error(y_true, y_pred)).item()

def mean_absolute_percentage_error(y_true : np.ndarray, y_pred :np.ndarray) -> float:
    ''' Returns Mean Absolute Percentage Error  = (1/n) Σ (|y_true(i) - y_pred(i)|/y_true) X 100'''
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred)/np.abs(y_true)).item() * 100

def r2_score(y_true : np.ndarray, y_pred :np.ndarray) -> float:
    ''' Returns R2 Score = 1- (Residual Sum of Squares/ Total Sum of Squares)'''
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot == 0:
        return np.nan
    r2 = 1 - (ss_res/ss_tot)
    return r2.item()

def score(y_true : np.ndarray, y_pred :np.ndarray) -> dict:
    ''' Returns dictionary of Evaluation Metrics of regression ie MSE, MAE, RMSE, R2, MAPE'''
    metrics = {
                    'MAE'   : mean_absolute_error(y_true, y_pred),
                    'MSE'   : mean_squared_error(y_true, y_pred),
                    'RMSE'  : root_mean_squared_error(y_true, y_pred),
                    'MAPE'  : mean_absolute_percentage_error(y_true, y_pred),
                    'R2'    : r2_score(y_true, y_pred)
                }
    return metrics

__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "score"
]
# Test Case 2: Typical Prediction
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

print(score(y_true, y_pred))
# Expected (approximately):
# MAE  = 0.5
# MSE  = 0.375
# RMSE = 0.612372
# MAPE = 32.7381
# R2   = 0.9486
