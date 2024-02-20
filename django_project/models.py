import numpy as np
import pandas as pd
import xgboost
import pickle as pkl
import os
import datetime
from django_project import data
from django.conf import settings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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


def scale_data(X, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X.copy())
    else:
        X = scaler.transform(X.copy())
    return X, scaler


def create_features(X, model, generation_type):
    # Create a list of all the variables to add to the model (not all of the base data can be used to predict with)
    base_features = ['Hour']
    # Create all the variables from combinations of all locations and hourly weather variables
    for i in settings.LOCATION_GEOCODE_DATA[generation_type].keys():
        for j in settings.HOURLY_WEATHER_VARIABLES:
            base_features.append(i+'_'+j)

    # create some feature within the dataframe
    X['DayofYear'] = X['Date'].dt.day_of_year
    # This assigns a weight to the day of year for the linear model to give most weight to middle of February
    X['DayofYearLr'] =  np.abs(45 - (np.abs(X['DayofYear'] - 45)) + 137)
    # This assigns a weight to the day of year day of year to give most weight to 21st June where the day is longer
    X['DayofYearAdj'] =  np.abs(172.25 - (np.abs(X['DayofYear'] - 172.25)) + 10)
    # This function is to simulate the number of hours of light in the day
    X['DayLength'] = 7.83 + 8.8 * np.sin(np.pi/2 * X['DayofYearAdj'] / 172.25)
    X = X[base_features+['DayofYear', 'DayofYearLr', 'DayofYearAdj', 'DayLength']]
    if model.__module__ == 'sklearn.linear_model._base':
        X = np.concatenate((X.values, create_lr_features(X[base_features])), axis=1)
    else:
        X = X.values
    return np.nan_to_num(X)


def show_training_score(generation_type, model, data, oob):
    data['oob'] = oob
    # Calculate the mean absolute error hourly
    hourly_mean_error = np.round(np.mean(np.abs(data['Power'] - oob)))
    # Calculate the mean absolute error at a daily level
    daily_data = data.groupby('Date').mean().reset_index()
    daily_mean_error = np.round(np.mean(np.abs(daily_data['Power'] - daily_data['oob'])))
    # Print the training errors
    print('Hourly mean error', hourly_mean_error, 'MW')
    print('Daily mean error', daily_mean_error, 'MW')
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(settings.DATA_DIR, 'model_revision_changelog.txt'), 'a') as f:
        f.write(ts + '\n')
        f.write(generation_type + ' ' + str(model.__module__) + '\n')
        f.write('Hourly mean error ' + str(hourly_mean_error) + ' MW\n')
        f.write('Daily mean error ' + str(daily_mean_error) + ' MW\n\n')
        f.close()
    return


# Cross-validate and fit predictive model
def cross_validate_and_fit_model(X, y, model, n_splits=5):
    print('\nTraining',model.__module__,'model')
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    scr = np.zeros([n_splits])
    oob = np.zeros(X.shape[0])
    for i, (tr_ix, val_ix) in enumerate(kf.split(X)):
        model.fit(X[tr_ix], y[tr_ix])
        pred = model.predict(X[val_ix])
        oob[val_ix] = np.array(pred)
        scr[i] = np.mean(np.abs(y[val_ix] - np.array(pred)))
    model.fit(X, y)
    return model, oob


def create_model(generation_type):
    if os.path.isfile(os.path.join(settings.DATA_DIR, 'ensemble_model_' + generation_type + '.pickle')):
        print("Saved model found")
        ensemble_model = pkl.load(open(os.path.join(settings.DATA_DIR, 'ensemble_model_'+generation_type+'.pickle'), "rb"))
        training_prediction = ensemble_model['data']
        scaler = ensemble_model['scaler']
        model_n = ensemble_model['model_n']
        model_a = ensemble_model['model_a']
        model_b = ensemble_model['model_b']
        model_c = ensemble_model['model_c']
        model_d = ensemble_model['model_d']
    else:
        # Test 3 types of predictive model
        model_n = MLPRegressor(hidden_layer_sizes=(256, 256), activation='relu', solver='adam', random_state=42, max_iter=6000, verbose=False, learning_rate_init=0.1, alpha=0.01)
        model_a = RandomForestRegressor(n_estimators=25, max_depth=12, min_samples_leaf=1)
        model_b = xgboost.XGBRegressor(tree_method="hist", eval_metric=mean_absolute_error, max_depth=12, min_child_weight=2)
        model_c = LinearRegression()
        model_d = xgboost.XGBRegressor(tree_method="hist", eval_metric=mean_absolute_error, max_depth=12, min_child_weight=2)
        
        base_data = data.read_training_data(generation_type)
        training_prediction = base_data.copy()[['Date', 'Hour', 'Power']]
        
        # Cross validate and fit each model on the training data
        model_list = [model_n, model_a, model_b, model_c]
        oob_features = np.zeros((len(base_data), len(model_list)))
        
        for i, model in enumerate(model_list):
            X = create_features(base_data, model, generation_type)
            if model == model_n:
                X, scaler = scale_data(X, None)
            y = base_data['Power'].to_numpy()
            model, oob = cross_validate_and_fit_model(X, y, model, n_splits=5)
            show_training_score(generation_type, model, base_data, oob)
            oob_features[:,i] = oob
            training_prediction.loc[:, 'Training_Prediction_' + str(i)] = np.clip(model.predict(X), 0 , 16000)
        
        # cv and create new model with the oob data added with XGBoost 
        X = create_features(base_data, model_d, generation_type)
        X = np.append(X, oob_features, axis=1)
        model, oob = cross_validate_and_fit_model(X, y, model_d, n_splits=5)
        show_training_score(generation_type, model_d, base_data, oob)
        
        training_prediction.loc[:, 'Training_Prediction_' + str(i)] = np.clip(model.predict(X), 0 , 16000)
        ensemble_model = {'data':training_prediction, 'scaler':scaler, 'model_n':model_n, 'model_a':model_a, 'model_b':model_b, 'model_c':model_c, 'model_d':model_d}
        pkl.dump(ensemble_model, open(os.path.join(settings.DATA_DIR, 'ensemble_model_'+generation_type+'.pickle'), 'wb'))
    
    return training_prediction, scaler, model_n, model_a, model_b, model_c, model_d
    

def save_forecast(forecast, generation_type):
    # The forecast data contains a mixture of historical data and data for the next few days
    # For the evaluation we only need to keep the forecasts on the future data (where the generation forecast might change as the weather forecast changes)
    forecast = forecast.copy()[forecast['Date']>=pd.Timestamp.today().floor('D')]
    forecast_columns = ['Created_at', 'Generation_type'] + list(forecast.columns)
    # Add a timestamp and the forecast type to the dataframe
    forecast['Created_at'] = pd.Timestamp.now()
    forecast['Generation_type'] = generation_type
    if os.path.isfile(os.path.join(settings.DATA_DIR, 'saved_forecasts.pickle')):
        print("Saved forecasts found")
        saved_forecasts = pkl.load(open(os.path.join(settings.DATA_DIR, 'saved_forecasts.pickle'), "rb"))
        saved_forecasts = pd.concat([saved_forecasts, forecast])
    else:
        print("Saving Forecast")
        saved_forecasts = forecast
    print(forecast)
    # Give the dataframe a better column order for reading and save the files
    saved_forecasts = saved_forecasts[forecast_columns]
    saved_forecasts.to_csv(os.path.join(settings.DATA_DIR, 'saved_forecasts.csv'), index=False)
    pkl.dump(saved_forecasts, open(os.path.join(settings.DATA_DIR, 'saved_forecasts.pickle'), "wb"))
    return


def evaluate_forecast(generation_data):
    if os.path.isfile(os.path.join(settings.DATA_DIR, 'saved_forecasts.csv')):
        print("Saved forecasts found")
        saved_forecasts = pkl.load(open(os.path.join(settings.DATA_DIR, 'saved_forecasts.pickle'), 'rb'))
    else:
        print("Missing forecast data")
    evaluation_data = saved_forecasts.merge(generation_data, on=["date", 'hour'])  
    return


def create_forecast(generation_type):
    forecast_file = os.path.join(settings.DATA_DIR, 'forecast_' + generation_type + '.pickle')
    # Load the current forecast file if there is one, and it has been recently created
    if os.path.isfile(forecast_file) and (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(forecast_file))).seconds < 1:
        print(generation_type, "forecast found")
        forecast = pkl.load(open(forecast_file, 'rb'))
    else:
        forecast_data = data.read_forecast_data_old(generation_type)
        forecast = forecast_data.copy()[['Date', 'Hour']]
        training_prediction, scaler, model_n, model_a, model_b, model_c, model_d = create_model(generation_type)
        # Make a prediction on the forecast data with each model
        print('\nMaking forecast with trained models')
        for j, model in enumerate([model_n, model_a, model_b, model_c]):
            # Make a prediction on the forecast data with each model
            X = np.nan_to_num(create_features(forecast_data, model, generation_type))
            if model == model_n:
                X, scaler = scale_data(X, scaler)
            forecast['Forecast_' + str(j)] = model.predict(X)
        
        # Add the forecasts for the other models to the training data and create a forecast with the stacked model
        X = create_features(forecast_data, model_d, generation_type)
        X = np.append(X, forecast[['Forecast_0', 'Forecast_1', 'Forecast_2', 'Forecast_3']].values, axis=1)
        forecast['Forecast_Stack'] = model_d.predict(X)
        
        # Create Ensemble Forecast
        forecast['Forecast_Ensemble'] = forecast['Forecast_0'] * 0.4 + \
                                        forecast['Forecast_1'] * 0.4 + \
                                        forecast['Forecast_2'] * 0.2
        
        pkl.dump(forecast, open(forecast_file, 'wb'))
        save_forecast(forecast, generation_type)
    
    return forecast
