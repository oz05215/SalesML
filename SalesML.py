import streamlit as st
import pandas as pd
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import numpy as np
#from fbprophet.plot import plot_plotly
from prophet.plot import plot_plotly
from io import BytesIO
import xlsxwriter
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

#



st.set_page_config(page_title = 'Sales Predictor',)
#st.set_page_config(page_title = 'Sales Predictor',layout='wide')






#User authentication

names = ['OscarZeledon','DiegoR']
usernames = ['Oskir','DiegoR']
#passwords = ['Cestern22','12345']



#load hashed passwords
file_path = Path(__file__).parent / 'Hashed_pw_Sales.pkl'
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)
   
#delete_all_cookies()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,'sales', 'abcdef', cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login","sidebar")



if authentication_status == False:
    st.error('Username/Password is incorrect')
    
if authentication_status == None:
    st.error('Please enter your username and password')
    
if authentication_status == True:

#sidebar

    with st.spinner("Generating Report"):   

    

#################################


        
        
        st.title('Sales Predictor')
        
            
        
        #text input
        st.markdown('---')
        st.subheader('Select the store code from 1 to 5')
        numero_de_tienda = st.number_input('Input Store id', min_value=0,max_value=3000, value=0, step=1)
        #describe = st.text_area(height=150)
        
        #Slider
        
        st.markdown('---')
        st.subheader('Number of days to predict')
        dias_a_predecir = st.slider(
            'Move the slider until reaching the desired amount of days to be predicted',0,365,0,5
            )
        
        st.write('Total number of days to be predicted',dias_a_predecir,' days') 
        
        ###################################################################################
        
        
        
        
        st.markdown('---')
        st.header('Historical file')
        st.subheader('Upload the most recent historical information')
        st.caption('Note: Predictions will start from the latest date in the document')
        
        uploaded_file2 = st.file_uploader('Choose a file in .csc or xlsx format:')
        save_button2 = st.button('Process')
        global sales_train_df
        
        if save_button2:
            if uploaded_file2 is not None:
                global sales_train_df
                st.success('File is being processed')
                sales_train_df = pd.read_csv(uploaded_file2)   
                   
                #global sales_train_df
                
                
                sales_train_df = sales_train_df[sales_train_df['Open']==1]
                
                #############################################################################################################################################################
                #############################################################################################################################################################
                #TRAIN INFO
                
                
                #keep only open stores
                sales_train_df = sales_train_df[sales_train_df['Open']==1]
                sales_train_df.count()
                
                #Drop open column
                
                sales_train_df.drop(['Open'], axis=1, inplace=True)
                
        
                
                tienda = sales_train_df[sales_train_df.Store==numero_de_tienda]
                st.write('### Statistics for Store id: ',numero_de_tienda)
        #        st.write(tienda.describe())
                
        

                #############################################################################################################################################################
        ######################################    
                
        
                store_info_df = pd.read_csv('store.csv')
        
                ##########################################################################################################################################################################################################################################################################################################################
                #############################################################################################################################################################
                #STORE INFO
                
                
                # When Promo2 is zero, then turn all the rest information related to zero
                
                str_cols = ['Promo2SinceWeek', 'Promo2SinceYear','PromoInterval', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']
                for str in str_cols:
                    store_info_df[str].fillna(0,inplace=True)
                    
                # Replace missing Distance data with the mean of the rest of the fields
                
                store_info_df['CompetitionDistance'].fillna(store_info_df['CompetitionDistance'].mean(),inplace=True)
                
                
                
                sales_train_all_df = pd.merge(sales_train_df, store_info_df, how='inner', on = 'Store')
                
                tiendados = sales_train_all_df[sales_train_all_df.Store==numero_de_tienda]
                
                #Find correlations
                st.write('##### Variables correlation')
                st.write('##### Near 0, no correlation with sales')
                st.write('##### Near 1, high correlation with sales')
                
                st.table(((sales_train_all_df.corr()['Sales'].sort_values(ascending=False))))
#                st.map(sales_train_all_df.corr()['Sales'])

                
                # SEPARATE THE YEAR FROM DATE
                
                sales_train_all_df['Year']= pd.DatetimeIndex(sales_train_all_df['Date']).year
                sales_train_all_df['Month']= pd.DatetimeIndex(sales_train_all_df['Date']).month
                sales_train_all_df['Day']= pd.DatetimeIndex(sales_train_all_df['Date']).day
                
                tiendados['Year']= pd.DatetimeIndex(tiendados['Date']).year
                tiendados['Month']= pd.DatetimeIndex(tiendados['Date']).month
                tiendados['Day']= pd.DatetimeIndex(tiendados['Date']).day
                
                
                #PLOT THE SALES AND CUSTOMERS BY MONTH
                
                
                axis = tiendados.groupby('Month')[['Sales']].mean()
                st.write('### Monthly average:')
                st.line_chart(axis)

                
                
                #PLOT THE SALES AND CUSTOMERS BY DAY

                st.session_state.axiss = tiendados.groupby('Day')[['Sales']].mean()
                st.write('### Daily average')
                st.line_chart(st.session_state.axiss)
                
                
                #PLOT THE SALES AND CUSTOMERS BY WEEKDAY
                
                
                axis = tiendados.groupby('DayOfWeek')[['Sales']].mean()
                st.write('### Weekday average')
                st.bar_chart(axis)
                
        
        
                import logging
                logging.getLogger('prophet').setLevel(logging.WARNING)
                
                
#                from fbprophet import Prophet
                from prophet import Prophet
                
                def sales_predictions(Store_ID, sales_df, holidays, periods):
                    sales_df = sales_df[sales_df['Store'] == Store_ID]
                    sales_df = sales_df[['Date','Sales']].rename(columns = {'Date':'ds', 'Sales':'y'})
                    sales_df = sales_df.sort_values('ds')
                    
                    model = Prophet(holidays = holidays)
                    model.fit(sales_df)
                    future = model.make_future_dataframe(periods = periods)
                    forecast = model.predict(future)
                    #figure = model.plot(forecast, xlabel ='Date', ylabel = 'Sales')
                    st.write('## Historical and Forecast Data')
                    st.write('###### Select in slidebar below the range desired')
                    fig1 = plot_plotly(model,forecast)
                    st.plotly_chart(fig1)
                    st.write('## Forecast Components')
                    figure2 = model.plot_components(forecast)
                    st.write(figure2)          
                    
        
                    max_date = pd.to_datetime(max(sales_df['ds']))
                    min_date = pd.to_datetime(min(sales_df['ds']))
                    initial_range = (max_date - min_date) / np.timedelta64(1, 'D')
                    initial_range = (int(int(initial_range) / 365))*365
                    
                    
                    from prophet.diagnostics import cross_validation
                    df_cv = cross_validation(model, initial= f'{initial_range} days', period='30 days', horizon = f'{dias_a_predecir} days')
        #            st.write(df_cv)
                    
                    
                    from prophet.diagnostics import performance_metrics
                    df_p = performance_metrics(df_cv)
                    df_p = pd.DataFrame(df_p)
                    df_p = df_p[['mdape']].mean()
                    percentage = "{:.2%}".format(df_p['mdape'])
                    st.write('## Mean Absolute Percentage Error (MAPE) is: ',percentage)
                    
                    
                    print(df_p)
        #            st.table(df_p)
                    from prophet.plot import plot_cross_validation_metric
                    fig = plot_cross_validation_metric(df_cv, metric='mape')
                    st.write(fig)
        
        
                    def to_excel(df):
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        df.to_excel(writer, index=False, sheet_name='Sheet1')
                        workbook = writer.book
                        worksheet = writer.sheets['Sheet1']
                        format1 = workbook.add_format({'num_format': '0.00'}) 
                        worksheet.set_column('A:A', None, format1)  
                        writer.save()
                        processed_data = output.getvalue()
                        return processed_data
                    df_xlsx = to_excel(forecast)
                    st.download_button(label='Save prediction',
                                                    data=df_xlsx ,
                                                    file_name= f'Prediction for {dias_a_predecir} days.xlsx')      
                
              
                
                school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1].loc[:,'Date'].values
        

        
                state_holidays = sales_train_all_df[(sales_train_all_df['StateHoliday'] == 'a') | (sales_train_all_df['StateHoliday'] == 'b')].loc[:,'Date'].values
                
                
                
                school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                               'holiday': 'school_holiday'})
                
                
                state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                                               'holiday': 'state_holiday'})
                
                
                school_state_holidays = pd.concat((state_holidays, school_holidays))
                
         #       daily_seasonality=True
                
                sales_predictions(numero_de_tienda, sales_train_all_df,school_state_holidays,dias_a_predecir)
                

        
        authenticator.logout('Logout','sidebar')
        
                
                
