import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('Pima Indian Diabetes Prediction App')
    st.sidebar.header('User Input Features')

    # Collect user input features
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    # Add more features as needed...

    # Make predictions
    input_data = {'Pregnancies': pregnancies, 'Glucose': glucose}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.subheader('Prediction')
    if prediction[0] == 0:
        st.write('No diabetes')
    else:
        st.write('Diabetes')

if __name__ == '__main__':
    main()
