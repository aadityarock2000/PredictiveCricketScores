import streamlit as st
import numpy as np
import pandas as pd
import gzip, pickle
from sklearn.preprocessing import StandardScaler
from joblib import load
from PIL import Image

st.set_page_config(
page_title="Predictive Cricket Score Analytics",
page_icon="🏏",
layout="centered",
)

dataset = pd.read_csv('data/final_PreProcess10.csv')

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(round(y_pred[i])-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)





st.title('Predictive Cricket Score Analytics')

#main page image
st.markdown("""
<img src="https://images.unsplash.com/photo-1607734834519-d8576ae60ea6" width="100%">""", unsafe_allow_html=True)
st.caption('A packed cricket statium in India')

tabs=st.tabs(['Introduction','Model Demo','Results'])
#Introduction to IPL

with tabs[0]:
    st.header('Introduction')
    col_1a,col_1b=st.columns(2)

    image = Image.open('assets/ipl_streamlit.jpg')
    col_1a.image(image, caption='IPL Team Captains and their different jerseys')
    
    # col_1a.markdown("""
    # <img src="assets/ipl_streamlit.jpg" width="100%">""", unsafe_allow_html=True)
    # col_1a.caption('IPL Team Captains and their different jerseys')

    col_1b.markdown("""
    I love watching the IPL (Indian Premier League). For the uninitiated, it is the most hyped sporting event
    in India, where 8-10 cricket leagues compete with each outher for the trophy. It consists of international stars 
    in Cricket from all over the world. The IPL is the most-attended cricket league in the world.
    
    """)

    st.markdown("""

    It is a professional men's Twenty20 cricket league, contested by ten teams based out of 
    seven Indian cities and three Indian states.The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. 
    It is usually held between March and May of every year and has an exclusive window in the ICC Future Tours Programme.

    Duing the first innings(play by the first team), there is always a discussion on the final score
    set by the team, which the second team should chase. This is most often calcualted naïvely using the
    formula shown below:
    """)
    st.latex(r"""
    current \space runs + current \space run \space rate * (remaining \space overs)
    """)

    st.markdown("""
    There is nothing inherently wrong with this approach, but this has a few disadvantages:
    1. It is highly fluctuating and can sometimes result in drastic numbers if a single batsman is playing extremely well at the moment.
    2. It doesn't take into account of the pace of the match. Generally, teams score anywhere between 150-200 runs on average, and this doesn't use this data to correct any mistakes.
    3. It ignores the results of the previous matches. History of the matches played would definitely be an indicator of what the current match's final score would look like (**wink! *Machine learning* Use case wink!**)

    Hence, I planned to use the data from the IPL matches played in the previous years to create a ML model to predict the final
    scores of the teams in the first innings.

    You can look at the demo of the model in the **Model Demo** tab and look at the results in the **Results** tab

    You can also loot at my [GitHub](https://github.com/aadityarock2000/PredictiveCricketScores) for more information on the process of building the machine learning model.
    """)

with tabs[1]:
    st.header('Model Demo')

    st.markdown("""
    <img src="https://images.unsplash.com/photo-1504639725590-34d0984388bd" width="100%">""", unsafe_allow_html=True)

    st.write(" ")


    #Features required:
    #total runs as of now - numerical input
    #total number of wickets fallen as of now - numerical input
    #ball(over number)[0.1-20) - numerical input
    #score of striker 
    #score of non striker


    col1,col2=st.columns(2)
    runs_tot=col1.number_input('Total runs as of now',step=1,format='%d')
    wickets=col2.number_input('Total number of wickets fallen as of now',step=1,format='%d')
    ball=col1.number_input('The ball(in terms of over) from 0.1 to 19.6',step=0.1,format='%f')
    striker=col2.number_input('Score of striker',step=1,format='%d')
    non_striker=col1.number_input('Score of non striker',step=1,format='%d')

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
        st.write('The final score predicted is:')
        st.write(output)

with tabs[2]:
    st.write("Coming Soon!")