import streamlit as st
import pandas as pd
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import numpy as np
import matplotlib.pyplot as plt
#from fbprophet.plot import plot_plotly
from prophet.plot import plot_plotly
from io import BytesIO


st.set_page_config(page_title = 'Sales Predictor')
#st.set_page_config(page_title = 'Sales Predictor',layout='wide')

    
import pickle
from pathlib import Path
import streamlit_authenticator as stauth




    
#for item in st.session_state.items():
#    item

#User authentication

names = ['OscarZeledon','DaniBlanco']
usernames = ['Oskir','Dblanco']
#passwords = ['Cestern22','12345']



#load hashed passwords
file_path = Path(__file__).parent / 'Hashed_pw.pkl'
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)
   
#delete_all_cookies()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,'sales_dashboard', 'abcdef', cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("Login","sidebar")

#if authentication_status not in st.session_state:
#    authentication_status = st.session_state.authentication_status

if authentication_status == False:
    st.error('Username/Password is incorrect')
    
if authentication_status == None:
    st.error('Please enter your username and password')
    
if authentication_status == True:

#sidebar

    with st.spinner("Generating Report"):   

#        upload_file = st.file_uploader("upload .csv or .xlsx files not exceeding 200 mb")


        

#    if upload_file is not None:
        #time bing let load csv
#        try:
#            df = pd.read_excel(upload_file)
#        except:
#            df = df = pd.read_csv(upload_file)
        
        #generate report
#        with st.spinner("Generating Report"):
#            pr = ProfileReport(df)
            
#        st_profile_report(pr)
        
#################################


        
        
        st.title('Predicción de ventas')
        
        #@st.cache(allow_output_mutation=True)
        #def load_model(model_name):
        #    if uploaded_file2 is not None:
        #        with open(os.path.join(r'C:\Users\oscar\Desktop\Aden Business School\TFM\Streamlit',uploaded_file2.name), mode='wb') as f:
        #            f.write(uploaded_file2.getbuffer())
        #        global sales_train_df
        #        sales_train_df = pd.read_csv(uploaded_file2)
        #        return sales_train_df
            
        
        #text input
        st.markdown('---')
        st.subheader('Seleccione el codigo de la tienda de la cual desea generar la prediccion')
        numero_de_tienda = st.number_input('Ingrese el identificador de la tienda', min_value=0,max_value=3000, value=0, step=1)
        #describe = st.text_area(height=150)
        
        #Slider
        
        st.markdown('---')
        st.subheader('Cantidad de días a predecir')
        dias_a_predecir = st.slider(
            'Mueva la barra hasta que obtenga la cantidad de dias que desea predecir en el futuro',0,365,0,5
            )
        
        st.write('La cantidad de dias a predecir desde la ultima fecha del archivo son',dias_a_predecir,' días') 
        
        ###################################################################################
        
        
        
        
        st.markdown('---')
        st.header('Documento de entrenamiento')
        st.subheader('Suba el documento que contiene el historial de ventas con la informacion mas reciente')
        st.caption('Nota: Las predicciones futuras se generaran empezando desde la ultima fecha que contenga el archivo')
        
        uploaded_file2 = st.file_uploader('Elija un documento:')
        save_button2 = st.button('Procesar archivo')
        global sales_train_df
        
        if save_button2:
            if uploaded_file2 is not None:
                global sales_train_df
                st.success('Archivo procesado exitosamente')
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
                st.write('### Estadisticas para la tienda #: ',numero_de_tienda)
        #        st.write(tienda.describe())
                
        
                
        #    else:
        #        st.warning('Porfavor seleccione el documento que desea subir')
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
                st.write('### Correlacion entre las variables')
                st.write('### Cercano a 0, no hay correlacion con los resultados de las ventas')
                st.write('### Cercano a 1, mucha correlacion con los resultados de las ventas')
                
                st.table(((sales_train_all_df.corr()['Sales'].sort_values(ascending=False))))
                
                
                # SEPARATE THE YEAR FROM DATE
                
                sales_train_all_df['Year']= pd.DatetimeIndex(sales_train_all_df['Date']).year
                sales_train_all_df['Month']= pd.DatetimeIndex(sales_train_all_df['Date']).month
                sales_train_all_df['Day']= pd.DatetimeIndex(sales_train_all_df['Date']).day
                
                tiendados['Year']= pd.DatetimeIndex(tiendados['Date']).year
                tiendados['Month']= pd.DatetimeIndex(tiendados['Date']).month
                tiendados['Day']= pd.DatetimeIndex(tiendados['Date']).day
                
                
                #PLOT THE SALES AND CUSTOMERS BY MONTH
                
                axis = sales_train_all_df.groupby('Month')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'black')
                
                axis.set_title('Average Sales per Month')
                
                plt.figure()
                
                axis = tiendados.groupby('Month')[['Sales']].mean()
                st.write('### Promedio en ventas por numero de mes para la tienda # :', numero_de_tienda)
                st.line_chart(axis)
                
                axis = sales_train_all_df.groupby('Month')[['Customers']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
                axis.set_title('Average Customers Per Month')
                
                
                #PLOT THE SALES AND CUSTOMERS BY DAY
                
                axis = sales_train_all_df.groupby('Day')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'black')
                
                st.session_state.axiss = tiendados.groupby('Day')[['Sales']].mean()
                st.write('### Promedio de ventas por dia del mes para la tienda #: ', numero_de_tienda)
                st.line_chart(st.session_state.axiss)
                
                axis.set_title('Average Sales per Day')
                
                plt.figure()
                
                axis = sales_train_all_df.groupby('Day')[['Customers']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
                axis.set_title('Average Customers Per Day')
                
                
                #PLOT THE SALES AND CUSTOMERS BY WEEKDAY
                
                axis = sales_train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'black')
                
                axis.set_title('Average Sales per Weekday')
                
                plt.figure()
                
                axis = tiendados.groupby('DayOfWeek')[['Sales']].mean()
                st.write('### Promedio en ventas por numero de dia de la semana para la tienda # :', numero_de_tienda)
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
                    st.write('## El porcentaje del promedio de error absoluto (MAPE) es: ',percentage)
                    
                    
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
                    st.download_button(label='Guardar Historico con predicción',
                                                    data=df_xlsx ,
                                                    file_name= f'Historico-Prediccion {dias_a_predecir} días.xlsx')      
                
              
                
                school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1].loc[:,'Date'].values
        
        
        
        
        
        
                state_holidays = sales_train_all_df[(sales_train_all_df['StateHoliday'] == 'a') | (sales_train_all_df['StateHoliday'] == 'b')].loc[:,'Date'].values
                
                
                
                school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                               'holiday': 'school_holiday'})
                
                
                state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                                               'holiday': 'state_holiday'})
                
                
                school_state_holidays = pd.concat((state_holidays, school_holidays))
                
         #       daily_seasonality=True
                
                sales_predictions(numero_de_tienda, sales_train_all_df,school_state_holidays,dias_a_predecir)
                
        
        #        try:
        #            if authenticator.reset_password(username, 'Reset password', 'sidebar'):
        #                st.success('Password modified successfully')
        #        except Exception as e:
        #            st.error(e)
                
        #RESET PASSWORD WIDGET
        
        authenticator.logout('Logout','sidebar')
        
                
                
