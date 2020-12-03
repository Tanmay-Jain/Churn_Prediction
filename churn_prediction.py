# import packages 
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

from class_balancing import balance_class_upsampling
from preprocessing import data_preparation

# predicting churn 
target_col = 'Churn' 

# Reading data 
data = pd.read_csv("Churn.csv")

# data pre-processing 
final_data = data_preparation(data)

# class balancing 
df_upsampled = balance_class_upsampling(final_data,target_col)

# Seperate Features and target column in X and y respectively. 
X = df_upsampled.drop(columns=[target_col])
y = df_upsampled[target_col]

# data splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# load xgboost pre-trained model 
model = joblib.load('churn_model.pkl')

# make prediction
prediction = model.predict(X_test)

output = X_test.copy()
output['Prediction'] = prediction
output.to_csv('Predictions.csv',index=False)
# save the model to disk
#filename = 'churn_model.pkl'
#joblib.dump(xgboost, open(filename, 'wb'))
