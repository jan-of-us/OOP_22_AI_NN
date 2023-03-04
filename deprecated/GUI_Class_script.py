import streamlit as st
import numpy as np
from numpy import random
import pandas as pd
import plotly_express as px

class GUI_class :
    '''
    Class for Primary dataset
    '''

    def __init__(self, arg_df, arg_filename='') :
        self.data = arg_df
        self.filename = arg_filename

        self.elements = self.data.size
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.nans = self.data.isna().sum().sum()
        
    
    def return_data(self) :
        return self.data



    def print_results(self) :
        st.markdown("# ")

        st.dataframe(self.data, use_container_width = True)

        num_elements, num_rows, num_cols, num_nans, download_data = st.columns(5)

        num_elements.metric('Elements', self.elements)
        num_rows.metric('Rows', self.rows)
        num_cols.metric('Columns', self.cols)
        num_nans.metric('Missing (NaN) Values', self.nans)

        download_data.download_button(label = 'Download Data', help = 'Download Dataframe as CSV File', data = self.data.to_csv(), mime = 'text/csv')

        if self.data.isnull().values.any() :
            self.null_and_index()



    def showInfo(self) :
        if self.filename == 'Dataset on Divorce' :
            self.showInfoDivorce()

        elif self.filename == 'Dataset on Energy' :
            self.showInfoEnergy()



    def showInfoDivorce(self) :
        st.markdown("# ")

        st.markdown("<h2 style = 'text-align : left; color : #0096c7;'> About The Data </h2>", unsafe_allow_html=True)

        st.markdown("<div style ='text-align: justify;'> Based on the <b>Gottman Couples Therapy</b>, couples were asked, how much they agree or disagree with 54 different questions (the attributes of the dataset), based on the state of their relationship. <br><br>The answers could range from :  </div>", unsafe_allow_html = True)

        st.markdown(""" 

            - **0** : Completely Disagree
            - **1** : Disagree
            - **2** : Neutral
            - **3** : Agree
            - **4** : Completely Agree

            """)

        st.markdown(" <div style ='text-align: justify;'> <br>Among the 170 participants interviewed, 86 belonged to <b>Class 0</b> (Married Couples) and 84 belonged to <b>Class 1</b> (Divorced Couples). The divorced participants were told to consider their past marriage when answering the questions. Only married participants who considered themselves happy, and had never thought of divorce, were included in the study.<br><br>The 54 attributes in the dataset were the ones remaining with a factor load of .40 or higher after performing an Exploratory Factor Analysis. Data collection was performed using face-to-face interviews or digitally. The data could then be used to find patterns to predict whether a couple would split up in the future or not.</div>", unsafe_allow_html = True)


        choice = st.selectbox(label = '' , options = ['Further Information :', 'About The Experimental Group', 'About The Source', 'About The Scientific Background Of The Study'])


        if choice == 'About The Experimental Group' :
            st.markdown(""" 

                - **Region** : Turkey, Europe

                - **170 Participants** : 
                    - 49% Divorced, 51% Married 
                    - 49% Male, 51% Female 
                    - 43.5% Married for Love (74), 56.5% Arranged Marriage (96)
                    - 74.7% Have Children (127), 25.3% Had No Children (43)
                    - **Age** : 20 to 63 (*Arithmetic Average : 36.04, Std. Deviation : 9.34*)  
                
                """)

            st.markdown(""" 

                - **Education** : 
                    - 10.58% were Primary School Graduates (18) 
                    - 8.8% were Secondary School Graduates (15) 
                    - 19.41% were High School Graduates (33) 
                    - 51.76% were College Graduates (88) 
                    - 8.8% had a Master’s Degree (15)

                """)

            st.markdown(""" 

                - **Monthly Income** : 
                    - 20%  had a monthly income *below 2,000 TL* (34) 
                    - 31.76%  had their monthly incomes between *2,001 and 3,000 TL* (54) 
                    - 16.47%  had their monthly incomes between *3,001 and 4,000 TL* (28)
                    - 31.76%  had a monthly income *above 4,000 TL* (54)
                
                <small> <b>*In 2018, 1 TL was roughly between 15 and 27 US$ Cents </b> <i>(Source: Google Finanzen)</i> </small> 
                
                """, unsafe_allow_html = True)


        elif choice == 'About The Source' :
            url = "https://dergipark.org.tr/en/pub/nevsosbilen/issue/46568/549416"

            st.markdown(""" 

                - **Institution :** Nevsehir Haci Bektas Veli University, SBE Dergisi, Turkey

                - **Introduction :** International Congress on Politics, Economics and Social Studies, 2018

                - **Research Article :** [Divorce Prediction using Correlation based Feature Selection and Artificial Neural Networks](%s) <small>(E-ISSN : 2149-3871), 2019</small>

                - **Authors :**
                    - Dr. Ögr. Üyesi Mustafa Kemal Yöntem
                    - Dr. Ögr. Üyesi Kemal Adem
                    - Prof. Dr. Tahsin Ilhan
                    - Ögr. Gör. Serhat Kilicarslan 

                """  % url, unsafe_allow_html = True)


        elif choice == 'About The Scientific Background Of The Study' :
            st.markdown(" <div style ='text-align: justify;'>The <b>Gottman Couples Therapy</b> modelled the reasons for divorce in married couples. Over time, the studies related to this model identified key factors that could cause divorce and showed accuracy when predicting, if a marriage will be long lasting or not. The <b>Divorce Predictors Scale <i>(DPS)</i></b> was developed by <b>Yöntem</b> and <b>Ilhan</b> and is based on the research done by Gottman.<br><br></div>", unsafe_allow_html = True)

            st.markdown(" <div style ='text-align: justify;'>The research group of the study decided to use <b>data mining technologies to predict the possibility of a divorce</b>. Data Mining had been successfully used in other fields of psychology and psychiatry. However, it had not yet been thoroughly used for divorce prediction. The aim of the study was <b>to contribute to the prevention of divorces by predicting them early</b>. Another target of the study was <b>to identify the most significant factors in the DPS that influenced the possibility of a divorce</b>. <br><br></div>", unsafe_allow_html = True)

            st.markdown(""" 

                The team applied the following algorithms to analyse the success of the Divorce Predictors Scale : 
                - Multilayer Perceptron Neural Network 
                - C4.5 Decision Tree algorithms

                """, unsafe_allow_html = True)

            st.markdown(" <div style ='text-align: justify;'><br>The best results were reached using an <b>Artificial Neural Net (ANN)</b> algorithm after selecting the most important 6 questions by applying the correlation-based feature selection method. Overall, the divorce predictors taken from the Gottman couples therapy were confirmed for the Turkish sample group.  <br><br></div>", unsafe_allow_html = True)



    def showInfoEnergy(self) :
        st.write('hello world')



    def null_and_index(self) :
        st.markdown("# ")

        RemoveNull, ResetIndex = st.columns(2)

        with RemoveNull :
            agree_null = st.checkbox('Remove Missing Values')

            if agree_null :
                self.data = self.data.dropna()


        with ResetIndex :
            agree_index = st.checkbox('Reset Data Frame Index')

            if agree_index :
                self.data = self.data.reset_index(drop = True)


        if agree_null or agree_index :
            self.print_results()

