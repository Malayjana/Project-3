# -*- coding: utf-8 -*-


import pickle
import streamlit as st
import pandas as pd
import numpy as np
#from PIL import Image
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier





pickle_in= open("C:/Users/HP/OneDrive/Desktop/project2_in_anew-env/RFC.pkl","rb")
classifier=pickle.load(pickle_in)



def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    #image = Image.open('churn.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    #st.sidebar.image(image)

    if add_selectbox == "Online":
         st.info("Input data below")
         #Based on our optimal features selection
         account_length = st.sidebar.number_input("account_length")
         voice_mail_plan = st.sidebar.selectbox('voice_mail_plan',('1','0'))
         day_mins = st.sidebar.number_input("day_mins")
         evening_mins =st.sidebar.number_input("evening_mins")
         night_mins =st.sidebar.number_input("night_mins")
         international_mins = st.sidebar.number_input("international_mins")
         customer_service_calls =st.sidebar.number_input("customer_service_calls")
         international_plan =st.sidebar.selectbox('international_plan',('1','0'))
         day_calls = st.sidebar.number_input("day_calls")
         evening_calls = st.sidebar.number_input("evening_calls")
         night_calls = st.sidebar.number_input("night_calls")
         international_calls = st.sidebar.number_input("international_calls")
         total_charge = st.sidebar.number_input("total_charge")
         data = {
             'account_length':account_length,
             'voice_mail_plan':voice_mail_plan,
             'day_mins':day_mins,
             'evening_mins':evening_mins,
             'night_mins':night_mins,
             'international_mins':international_mins,
             'customer_service_calls':customer_service_calls,
             'international_plan':international_plan,
             'day_calls':day_calls,
             'evening_calls':evening_calls,
             'night_calls':night_calls,
             'international_calls':international_calls,
             'total_charge':total_charge
            }
           

         features_df = pd.DataFrame.from_dict([data])
         st.markdown("<h3></h3>", unsafe_allow_html=True)
         st.write('Overview of input is shown below')
         st.markdown("<h3></h3>", unsafe_allow_html=True)
         st.dataframe(features_df)



  

         prediction = classifier.predict(features_df)

         if st.button('Predict'):
            if prediction == 1:
                 st.warning('Unfortunately this person is going to churn.')
            else:
                 st.success('Congratulation this person is not going to Churn.')
                 st.balloons()
        

    else:
        st.subheader("Please upload Dataset file in csv format here seperated by Semicolons';'. ")
        uploaded_file = st.file_uploader("---")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)#, sep=';')
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            X = data.iloc[: , :-1]
            Y = data.iloc[:, -1]
            x_train,x_test,y_train,y_test = train_test_split(X,Y , test_size=0.30,random_state=0)
            train_scaler=StandardScaler()
            test_scaler=StandardScaler()
            x_train=train_scaler.fit_transform(x_train)
            x_test=test_scaler.fit_transform(x_test)
            ordered_rank_features=SelectKBest(score_func=chi2,k=18)
            ordered_features=ordered_rank_features.fit(X,Y)
            dfscores=pd.DataFrame(ordered_features.scores_,columns=['scores'])
            dfcolumns=pd.DataFrame(X.columns)
            features_rank=pd.concat([dfcolumns,dfscores],axis=1)
            features_rank.columns=['features','scores']
            features_rank.nlargest(14,'scores')
            corr=data.iloc[:,:-1].corr()
            
            threshold=0.9
            
            def corr(dataset,threshold):
                col_corr=set()
                corr_matrix=data.corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i,j])>threshold:
                            colname=corr_matrix.columns[i]
                            col_corr.add(colname)
                return col_corr
            corr_featues = corr(data,threshold)
            corr_featues
            st.write('These are Most correlated features')
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            data1=data.drop(corr_featues,axis=1)
            st.write(data1.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            x = data1.iloc[: , :-1]
            y = data1.iloc[:, -1]
            x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)
            upsample = SMOTE()
            x_train, y_train = upsample.fit_resample(x_train, y_train)
            y=pd.DataFrame(y_train)
            
            
            
            #Preprocess inputs

            if st.button('Predict'):
                RFC= RandomForestClassifier()
                RFC.fit(x_train,y_train)
                prediction = RFC.predict(x)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Unfortunately, this person is going to churn.', 
                                                    0:'Congratulation, this person is not going to Churn.'})
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
        

