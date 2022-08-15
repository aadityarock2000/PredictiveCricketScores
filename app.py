import streamlit as st
import numpy as np
import pandas as pd
import gzip, pickle
from sklearn.preprocessing import StandardScaler
from joblib import load

dataset = pd.read_csv('data/final_PreProcess10.csv')

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(round(y_pred[i])-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)


st.title('Predictive Cricket Score Analytics')

st.header('Model Demo')
#Features required:
#total runs as of now - numerical input
#total number of wickets fallen as of now - numerical input
#ball(over number)[0.1-20) - numerical input
#score of striker 
#score of non striker
col1,col2=st.columns(2)
runs_tot=col1.number_input('Total runs as of now',step=1,format='%d')
wickets=col2.number_input('Total number of wickets fallen as of now',step=1,format='%d')
ball=col1.number_input('the ball(in terms of over) from 0.1 to 19.6',step=0.1,format='%f')
striker=col2.number_input('score of striker',step=1,format='%d')
non_striker=col1.number_input('score of non striker',step=1,format='%d')

if st.button('Predict the final Score'):   
    data=np.array([runs_tot,wickets,ball,striker,non_striker])
    data=data.reshape(1,-1)
    
    # Feature Scaling
    sc=load('std_scaler.bin')
    data=sc.transform(data)

    #applying the model to get the prediction
    filepath = "random_forest_optimised.pkl"
    model =' '
    with gzip.open(filepath, 'rb') as f:
        p = pickle.Unpickler(f)
        model = p.load()

    #model = pickle.load(open('random_forest_regressor.pkl', 'rb'))
    output= int(model.predict(data))
    st.write('The final Score predicted is:')
    st.write(output)
    #print(type(output))
