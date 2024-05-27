#Importing the neessary libraries
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
import mlflow

# #Setting mlflow tracking
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment("nyc_taxi")


# df = pd.read_parquet('./green_tripdata_2021-01.parquet')

# df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
# df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
# df = df[(df.duration >= 1) & (df.duration <= 60)]
# categorical = ['PULocationID', 'DOLocationID']
# numerical = ['trip_distance']
# df[categorical] = df[categorical].astype(str)


# train_dicts = df[categorical + numerical].to_dict(orient='records')
# dv = DictVectorizer()
# X_train = dv.fit_transform(train_dicts)
# target = 'duration'
# y_train = df[target].values
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_train)
# mean_squared_error(y_train, y_pred, squared=False)



# sns.distplot(y_pred, label='prediction')
# sns.distplot(y_train, label='actual')
# plt.legend()


def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def add_features(train_path = "./green_tripdata_2021-01.parquet",val_path = "./green_tripdata_2021-02.parquet"):

   df_train = read_dataframe(train_path)
   df_val = read_dataframe(val_path)
   #len(df_train), len(df_val)

   df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
   df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
   categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
   numerical = ['trip_distance']

   dv = DictVectorizer()

   train_dicts = df_train[categorical + numerical].to_dict(orient='records')
   X_train = dv.fit_transform(train_dicts)

   val_dicts = df_val[categorical + numerical].to_dict(orient='records')
   X_val = dv.transform(val_dicts)


   target = 'duration'
   y_train = df_train[target].values
   y_val = df_val[target].values
   return X_train,X_val,y_train,y_val,dv

X_train,X_val,y_train,y_val,dv = add_features()
print(X_train.shape)

###########
# Modeling section
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# y_pred = lr.predict(X_val)

# mean_squared_error(y_val, y_pred, squared=False)

# import os

# # Create the directory if it doesn't exist
# os.makedirs('models', exist_ok=True)

# # Save the model
# with open('models/lin_reg.bin', 'wb') as f_out:
#     pickle.dump((dv, lr), f_out)

# print("Model saved successfully!")

# with mlflow.start_run():
#     # Set tag for mlflow
#     mlflow.set_tag("developer", "Vijay")

#     # Set data path params
#     mlflow.log_param("train-data-path", './green_tripdata_2021-01.parquet')
#     mlflow.log_param("valid-data-path", './green_tripdata_2021-02.parquet')

#     # Log alpha parameter
#     alpha = 0.01
#     mlflow.log_param("alpha", alpha)

#     # Train the model
#     lr = Lasso(alpha)
#     lr.fit(X_train, y_train)

#     # Predict on validation data
#     y_pred = lr.predict(X_val)

#     # Calculate and log RMSE
#     rmse = mean_squared_error(y_val, y_pred, squared=False)
#     mlflow.log_metric("rmse", rmse)

# import xgboost as xgb

# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from hyperopt.pyll import scope

# train = xgb.DMatrix(X_train, label=y_train)
# valid = xgb.DMatrix(X_val, label=y_val)


# def objective(params):
#     with mlflow.start_run():
#         mlflow.set_tag("model", "xgboost")
#         mlflow.log_params(params)
#         booster = xgb.train(
#             params=params,
#             dtrain=train,
#             num_boost_round=1000,
#             evals=[(valid, 'validation')],
#             early_stopping_rounds=50
#         )
#         y_pred = booster.predict(valid)
#         rmse = mean_squared_error(y_val, y_pred, squared=False)
#         mlflow.log_metric("rmse", rmse)

#     return {'loss': rmse, 'status': STATUS_OK}



# search_space = {
#     'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
#     'learning_rate': hp.loguniform('learning_rate', -3, 0),
#     'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
#     'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
#     'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
#     'objective': 'reg:linear',
#     'seed': 42
# }


# best_result = fmin(
#     fn=objective,
#     space=search_space,
#     algo=tpe.suggest,
#     max_evals=20,
#     trials=Trials()
# )


# params = {
#     'max_depth': 4,
#     'learning_rate': 0.14493221791716185,
#     'reg_alpha': 0.012153110171030913,
#     'reg_lambda': 0.017881159785939696,
#     'min_child_weight': 0.674864917045824,
#     'objective': 'reg:squarederror',  # 'reg:linear' is deprecated, use 'reg:squarederror'
#     'seed': 42
# }

# mlflow.xgboost.autolog()

# booster = xgb.train(
#             params=params,
#             dtrain=train,
#             num_boost_round=1000,
#             evals=[(valid, 'validation')],
#             early_stopping_rounds=5
#         )


# mlflow.end_run()


# import xgboost as xgb
# # Disable autologging for this run
# mlflow.xgboost.autolog(disable=True)

# with mlflow.start_run():
#     # Hyperparameters for the run
#     params = {
#         'max_depth': 4,
#         'learning_rate': 0.14493221791716185,
#         'reg_alpha': 0.012153110171030913,
#         'reg_lambda': 0.017881159785939696,
#         'min_child_weight': 0.674864917045824,
#         'objective': 'reg:squarederror',  # Updated objective function
#         'seed': 42
#     }

#     # Log parameters to MLflow
#     mlflow.log_params(params)

#     # Train the model
#     booster = xgb.train(
#         params=params,
#         dtrain=train,
#         num_boost_round=1000,
#         evals=[(valid, 'validation')],
#         early_stopping_rounds=5
#     )

#     # Predict on validation data
#     y_pred = booster.predict(valid)

#     # Calculate and log RMSE to MLflow
#     rmse = round(mean_squared_error(y_val, y_pred, squared=False), 2)
#     print("RMSE for validation data:", rmse)
#     mlflow.log_metric("rmse", rmse)

#     # Log the XGBoost model to MLflow
#     mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

#     # Log the preprocessor (DictVectorizer) to MLflow
#     with open("models/preprocessor.bin", "wb") as f_out:
#         pickle.dump(dv, f_out)

#     mlflow.log_artifact("models/preprocessor.bin", artifact_path="preprocessor")


