# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: siddhardhan
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
import numpy as np

df1 = pd.read_csv('new.csv')

X = df1.drop(['price'], axis="columns")
y = df1.price



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)




import pickle

# Open the file in binary mode
with open('C:/Users/vkpat/Desktop/HOUSE/DATASER_/House-Price-Prediction-Django-ML/Server/artifacts/banglore_home_prices_model_random.pickle', 'rb') as file:
    
    # Call load method to deserialze
    myvar = pickle.load(file)


import json
columns = {
    'data_columns' : [col for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))




def predict_price(location,sqft):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    # x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return myvar.predict([x])[0]

n=predict_price('Vile Parle West',2500)
b=predict_price('Thane West',350)
g=predict_price('Taloja',350)
k=predict_price('Ghatkopar East',750)
st.success(n)
st.success(b)
st.success(g)
st.success(k)




# # creating a function for Prediction

# def diabetes_prediction(input_data):
    

#     # changing the input_data to numpy array
#     input_data_as_numpy_array = np.asarray(input_data)

#     # reshape the array as we are predicting for one instance
#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#     prediction = loaded_model.predict(input_data_reshaped)
#     print(prediction)

#     if (prediction[0] == 0):
#       return 'The person is not diabetic'
#     else:
#       return 'The person is diabetic'
  
    
  
def main():

    with open("columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    # st.text(__locations)
    st.selectbox("new",options=__locations)


    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  