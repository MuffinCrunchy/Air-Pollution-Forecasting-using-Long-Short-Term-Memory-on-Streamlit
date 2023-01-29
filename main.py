import yaml
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator

# Authentication
with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login Form
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == None:
    st.warning("Please enter your name an password")
elif authentication_status == False:
    st.error("Username/Password is Incorrect")
elif authentication_status:
    authenticator.logout("Logout", "sidebar")

    # Main Page
    
    # App Title
    st.title("Air Pollution Forecasting using Long Short-Term Memory")

    # Sidebar Title
    st.sidebar.title("Settings")
    
    # Datasets Settings
    st.sidebar.header("Datasets")

    def get_dataset(choose_data):
        if choose_data == 'Jakarta Pusat':
            df = pd.read_csv('./datasets/jkt-pusat-ispu-2011-2021-knn.csv')
        elif choose_data == 'Jakarta Utara':
            df = pd.read_csv('./datasets/jkt-utara-ispu-2011-2021-knn.csv')
        elif choose_data == 'Jakarta Barat':
            df = pd.read_csv('./datasets/jkt-barat-ispu-2011-2021-knn.csv')
        elif choose_data == 'Jakarta Selatan':
            df = pd.read_csv('./datasets/jkt-selatan-ispu-2011-2021-knn.csv')
        elif choose_data == 'Jakarta Timur':
            df = pd.read_csv('./datasets/jkt-timur-ispu-2011-2021-knn.csv')
        
        return df

    def get_pollutant(choose_poll):
        if choose_poll == 'PM10':
            pt = 'pm10'
        elif choose_poll == 'SO2':
            pt = 'so2'
        elif choose_poll == 'CO':
            pt = 'co'
        elif choose_poll == 'O3':
            pt = 'o3'
        elif choose_poll == 'NO2':
            pt = 'no2'
        
        return pt

    def get_chart(choose_chart):
        if choose_chart == "Lineplots":
            plot = px.line(data_frame=df, x=df['tanggal'], y=df[pt])
        elif choose_chart == "Scatterplots":
            plot = px.scatter(data_frame=df, x=df['tanggal'], y=df[pt])

        return plot

    def get_time(choose_time):
        if choose_time == '1 Year':
            df_time = df[3287:3653]
        elif choose_time == '5 Years':
            df_time = df[1826:3653]
        elif choose_time == '10 Years':
            df_time = df[:3653]

        df_test = df[3653:]
        df_time.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        return df_time, df_test

    def train_valid_split(choose_time):
        if choose_time == '1 Year':
            size = 334
        elif choose_time == '5 Years':
            size = 1461
        elif choose_time == '10 Years':
            size = 2922

        df_train = df_time[:size]
        df_train.reset_index(drop=True, inplace=True)
        df_valid = df_time[size:]
        df_valid.reset_index(drop=True, inplace=True)

        return df_train, df_valid

    def get_slm(norm_data, look_back, batch_size):
        slm_data = TimeseriesGenerator(norm_data, norm_data, length=look_back, batch_size=batch_size)

        return slm_data

    def set_model(lstm_1, lstm_2, lstm_3):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units = lstm_1, return_sequences = True, input_shape = (look_back, 1)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units = lstm_2, return_sequences = True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units = lstm_3))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(units = 1))

        return model

    def set_compiler(model):
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

        return model

    def compare_plot(df):
        df_1 = go.Scatter(
            x = df.index,
            y = df[df.columns[0]],
            mode = 'lines',
            name=df.columns[0]
        )
        df_2 = go.Scatter(
            x = df.index,
            y = df[df.columns[1]],
            mode = 'lines',
            name=df.columns[1]
        )
        layout = go.Layout()
        plot = go.Figure(data=[df_1, df_2], layout=layout)

        return plot



    # Select Dataset
    choose_data = st.sidebar.selectbox("Choose a Dataset", options=['Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur'])
    st.header(choose_data)
    st.subheader("Dataframe")
    df = get_dataset(choose_data)
    st.write(df)


    # Select Pollutant
    choose_poll = st.sidebar.selectbox("Select the Pollutant", options=['PM10', 'SO2', 'CO', 'O3', 'NO2'])
    st.header(choose_poll)
    pt = get_pollutant(choose_poll)
    df = df[["tanggal", pt]]


    # Select Chart
    choose_chart = st.sidebar.selectbox(label="Select the chart type", options = ["Lineplots", "Scatterplots"])
    st.subheader("Chart: {}".format(choose_chart))
    plot = get_chart(choose_chart)
    st.plotly_chart(plot)


    # Select Time Range
    choose_time = st.sidebar.selectbox("Select Time Range", options=["1 Year", "5 Years", "10 Years"])
    st.header("Time Range: {}".format(choose_time))
    df_time, df_test = get_time(choose_time)
    col0_1, col0_2 = st.columns(2)
    col0_1.write(df_time)
    col0_2.write(df_test)


    # Train-Valid-Test Split
    df_train, df_valid = train_valid_split(choose_time)

    st.header("Train-Validate-Test Split")
    col1_1, col1_2, col1_3 = st.columns(3)
    col1_1.subheader("Data Train")
    col1_1.write(df_train)
    col1_2.subheader("Data Validate")
    col1_2.write(df_valid)
    col1_3.subheader("Data Test")
    col1_3.write(df_test)


    # Normalization
    minmax = MinMaxScaler(feature_range=(0, 1))

    norm_train = minmax.fit_transform(df_train[[pt]])
    norm_valid = minmax.fit_transform(df_valid[[pt]])
    norm_test = minmax.fit_transform(df_test[[pt]])


    # Visualize Normalization
    df_norm_train = pd.DataFrame(norm_train, columns = [pt])
    df_norm_valid = pd.DataFrame(norm_valid, columns = [pt])
    df_norm_test = pd.DataFrame(norm_test, columns = [pt])

    df_norm_train = pd.concat([df_train[['tanggal']], df_norm_train], axis=1)
    df_norm_valid = pd.concat([df_valid[['tanggal']], df_norm_valid], axis=1)
    df_norm_test = pd.concat([df_test[['tanggal']], df_norm_test], axis=1)

    st.header("Normalized Data")
    col2_1, col2_2, col2_3 = st.columns(3)
    col2_1.subheader("Data Train")
    col2_1.write(df_norm_train)
    col2_2.subheader("Data Validate")
    col2_2.write(df_norm_valid)
    col2_3.subheader("Data Test")
    col2_3.write(df_norm_test)


    # SLM Settings
    st.sidebar.header("Supervised Learning Problem")

    # Select Look Back
    look_back = st.sidebar.slider('Timesteps', 1, 10, 1)
    
    # Select Batch Size
    st.sidebar.subheader("Batch Size")
    batch_size_train = st.sidebar.slider('Train', 1, 30, 30)
    batch_size_valid = st.sidebar.slider('Validate', 1, 7, 7)
    batch_size_test = st.sidebar.slider('Test', 1, 7, 1)

    # Supervised Learning Model
    slm_train = get_slm(norm_train, look_back, batch_size_train)
    slm_valid = get_slm(norm_valid, look_back, batch_size_valid)
    slm_test = get_slm(norm_test, look_back, batch_size_test)

    # Data Test for Predict
    for_predict = norm_test[look_back:]

    # Model Parameter Settings
    st.sidebar.header("Model Parameter")

    # Select Neurons
    st.sidebar.subheader("Number of Neurons")
    lstm_1 = st.sidebar.slider('First Layer', 16, 128, step=16)
    lstm_2 = st.sidebar.slider('Second Layer', 16, 128, step=16)
    lstm_3 = st.sidebar.slider('Third Layer', 16, 128, step=16)

    # Select Epochs
    st.sidebar.subheader("Number of Epochs")
    epochs = st.sidebar.slider('First Layer', 10, 50, step=10)

    # Initialize Model
    st.header("Architecture Model")
    model = set_model(lstm_1, lstm_2, lstm_3)
    model.summary(print_fn=lambda x: st.text(x))

    # Compile Model
    model = set_compiler(model)

    # Train Model
    st.header("Train Model")

    container = st.empty()
    placeholder = st.empty()
    btn = placeholder.button("Start Training", disabled=False, key='1')
    if btn:
        placeholder.button('Start Training', disabled=True, key='2')
        with st.spinner('Grab a cup of coffee or do something! This will take a while...'):
            history = model.fit(slm_train, validation_data=slm_valid, epochs = epochs)
        st.success("Your model is back from training. Look below to see the result.")

        # Result
        st.header("Result")

        # Loss Graphs
        st.subheader("Loss")
        loss = list(zip(history.history['loss'], history.history['val_loss']))
        df_loss = pd.DataFrame(loss, columns=["Loss", "Validation Loss"])
        plot_loss = compare_plot(df_loss)
        st.write(plot_loss)

        # MAE Graphs
        st.subheader("MAE")
        mae = list(zip(history.history['mae'], history.history['val_mae']))
        df_mae = pd.DataFrame(loss, columns=["MAE", "Validation MAE"])
        plot_mae = compare_plot(df_mae)
        st.write(plot_mae)

        # Test Model
        st.header("Test Model")
        norm_predict = model.predict_generator(for_predict)

        predict = minmax.inverse_transform(norm_predict)
        actual = minmax.inverse_transform(for_predict)

        # Visualize Prediction
        st.subheader("Result")
        forecast = list(zip(actual.reshape(-1), predict.reshape(-1)))
        df_forecast = pd.DataFrame(forecast, columns=['Actual', 'Prediction'])
        plot_forecast = compare_plot(df_forecast)
        st.write(plot_forecast)

        # RMSE
        st.subheader("Score")
        error_predict = predict.reshape((-1))
        error_actual = actual.reshape((-1))
        score = np.sqrt(mean_squared_error(error_actual, error_predict))
        st.write("RMSE Score: {0:.{1}f}".format(score, 2))
        
        # Reset Model
        placeholder.button('Reset Training', disabled=False, key='3')
        container.empty()

        
    
    


