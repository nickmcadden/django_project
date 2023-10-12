import numpy as np
import xgboost
import pickle as pkl
import os
from django_project import data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def transform_angle(a):
    a = a - 32
    transform = np.abs(90 - np.abs(180 - a))
    weight = np.sin(np.radians(transform))
    return weight


# Create some data transformations for the linear regression
def create_lr_features(X):
    f05 = np.power(X, 0.5)
    f08 = np.power(X, 0.8)
    f12 = np.power(X, 1.2)
    f16 = np.power(X, 1.6)
    f20 = np.power(X, 2)
    f24 = np.power(X, 2.4)
    f28 = np.power(X, 2.8)
    f30 = np.power(X, 3.0)
    wd_feature = np.multiply(transform_angle(X), np.power(X, 0.8))
    X = np.concatenate((X, f05, f08, f12, f16, f20, f24, f28, f30, wd_feature), axis=1)    
    return X


def create_features(X, model):
    # Input fields for the model
    base_features = ['Hour', 'Bracknell_windspeed_10m', 'Bracknell_winddirection_10m', 
                        'Cardiff_windspeed_10m', 'Cardiff_winddirection_10m',
                        'Leeds_windspeed_10m', 'Leeds_winddirection_10m',
                        'Belfast_windspeed_10m', 'Belfast_winddirection_10m',
                        'Edinburgh_windspeed_10m', 'Edinburgh_winddirection_10m',
                        'Inverness_windspeed_10m', 'Inverness_winddirection_10m',
                        'Norwich_windspeed_10m', 'Norwich_winddirection_10m',
                        'Hull_windspeed_10m', 'Hull_winddirection_10m',
                        'Carlisle_windspeed_10m', 'Carlisle_winddirection_10m']
    
    # create some feature within the dataframe
    X['DayofYear'] = X['Date'].dt.day_of_year
    X['DayofYearLr'] =  500 - (np.abs(X['DayofYear'] - 45))
    X = X[base_features+['DayofYear', 'DayofYearLr']]
    if model.__module__ == 'sklearn.linear_model._base':
        X = np.concatenate((X.values, create_lr_features(X[base_features])), axis=1)
    else:
        X = X.values
    return X


# Cross-validate and fit predictive model
def cross_validate_and_fit_model(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    scr = np.zeros([n_splits])
    oob = np.zeros(X.shape[0])
    for i, (tr_ix, val_ix) in enumerate(kf.split(X)):
        model.fit(X[tr_ix], y[tr_ix])
        pred = model.predict(X[val_ix])
        oob[val_ix] = np.array(pred)
        scr[i] = np.mean(np.abs(y[val_ix] - np.array(pred)))
        print('', end='>')
    print(' Training model mean error', np.round(np.mean(np.abs(y - oob))),'MW')
    model.fit(X, y)
    return model, oob


def create_model(): 
    if os.path.isfile("ensemble_model.pickle"):
        print("Saved model found")
        ensemble_model = pkl.load(open("ensemble_model.pickle", "rb"))
        training_prediction = ensemble_model['data']
        model_a = ensemble_model['model_a']
        model_b = ensemble_model['model_b']
        model_c = ensemble_model['model_c']
    else:
        # Test 3 types of predictive model
        model_a = RandomForestRegressor(n_estimators=25, max_depth=12, min_samples_leaf=1)
        model_b = xgboost.XGBRegressor(tree_method="hist", eval_metric=mean_absolute_error, max_depth=12, min_child_weight=2)
        model_c = LinearRegression()
        model_d = xgboost.XGBRegressor(tree_method="hist", eval_metric=mean_absolute_error, max_depth=12, min_child_weight=2)
        
        base_data = data.read_training_data(start_date='2022-08-01')
        training_prediction = base_data.copy()[['Date', 'Hour', 'Wind']]
        
        # Cross validate and fit each model on the training data
        model_list = [model_a, model_b, model_c]
        oob_features = np.zeros((len(base_data), len(model_list)))
        
        for i, model in enumerate(model_list):
            print('Training',model.__module__,'model')
            X = create_features(base_data, model)
            y = base_data['Wind'].to_numpy()
            model, oob = cross_validate_and_fit_model(X, y, model, n_splits=5)
            oob_features[:,i] = oob
            training_prediction.loc[:, 'Training_Prediction_' + str(i)] = np.clip(model.predict(X) ,1000 , 16000)
        
        # cv and create new model with the oob data added with XGBoost 
        X = create_features(base_data, model_d)
        X = np.append(X, oob_features, axis=1)
        model, oob = cross_validate_and_fit_model(X, y, model_d, n_splits=5)
        training_prediction.loc[:, 'Training_Prediction_' + str(i)] = np.clip(model.predict(X) ,1000 , 16000)
        ensemble_model = {'data':training_prediction, 'model_a':model_a, 'model_b':model_b, 'model_c':model_c, 'model_d':model_d}
        #pkl.dump(ensemble_model, open("ensemble_model.pickle", "wb"))
    
    return training_prediction, model_a, model_b, model_c, model_d
    

def create_forecast(forecast_data):
    training_prediction, model_a, model_b, model_c, model_d = create_model()
    forecast = forecast_data.copy()[['Date', 'Hour']]
    # Make a prediction on the forecast data with each model
    print('\nMaking forecast with trained models')
    for j, model in enumerate([model_a, model_b, model_c]):
        # Make a prediction on the forecast data with each model
        X = create_features(forecast_data, model)
        forecast['Forecast_' + str(j)] = np.clip(model.predict(X),1000,16000)
    
    # Add the forecasts for the other models to the training data and create a forecast with the stacked model
    X = create_features(forecast_data, model_d)
    X = np.append(X, forecast[['Forecast_0', 'Forecast_1', 'Forecast_2']].values, axis=1)
    forecast['Forecast_Stack'] = np.clip(model_d.predict(X),1000,16000)
    
    # Create Ensemble Forecast (Currently XGBoost results can't be improved with other model combination)
    forecast['Forecast_Ensemble'] = forecast['Forecast_0'] * 0.4 + \
                                    forecast['Forecast_1'] * 0.4 + \
                                    forecast['Forecast_2'] * 0.2
    
    return training_prediction, forecast
