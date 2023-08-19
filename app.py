# import json
# import pickle

# from flask import Flask,request,app,jsonify,url_for,render_template
# import numpy as np
# import pandas as pd

# app=Flask(__name__)
# ## Load the model
# regmodel=pickle.load(open('regmodel.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     # whenever i hit the /predict_api as a post request then input that we will give is going to be in the JSON format which will be captured inside the "data" key and its get stored inside the data variable
#     final_input=scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     # once we get the data it will be in a key value pair
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


# if __name__=="__main__":
#     app.run(debug=True)


import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Define feature descriptions
feature_descriptions = {
    "CRIM": "per capita crime rate by town",
    "ZN": "proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
    "NOX": "itric oxides concentration (parts per 10 million)",
    "RM": "average number of rooms per dwelling",
    "AGE": "proportion of owner-occupied units built prior to 1940",
    "DIS": "weighted distances to five Boston employment centres",
    "RAD": "index of accessibility to radial highways",
    "TAX": "full-value property-tax rate per $10,000",
    "PTRATIO": "pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is the proportion of black people by town",
    "LSTAT": "% lower status of the population",

    # Add more feature descriptions
}


def main():
    st.title("House Price Prediction")

    st.sidebar.header("Input Features")

    input_features = {}

    # Collect user input for each feature
    for feature_name in feature_descriptions.keys():
        description = feature_descriptions[feature_name]
        input_value = st.sidebar.text_input(
            f"{feature_name}: {description}", value="0.0")
        input_features[feature_name] = float(input_value)

    # Convert input features to a numpy array
    input_array = np.array(list(input_features.values())).reshape(1, -1)
    scaled_input = scalar.transform(input_array)

    # Make a prediction
    prediction = regmodel.predict(scaled_input)[0]

    # Display the prediction in a box
    st.success(f"The predicted house price is: {prediction}")
  # Display bar chart using Matplotlib
    st.subheader("Predicted Price Bar Chart")
    bar_data = {"Predicted Price": prediction}
    fig, ax = plt.subplots()
    ax.bar(bar_data.keys(), bar_data.values())
    ax.set_xlabel("Categories")
    ax.set_ylabel("Price")
    ax.set_title("Predicted Price Bar Chart")
    st.pyplot(fig)

    # Display line chart using Matplotlib
    st.subheader("Random Data Line Chart")
    random_data = np.random.rand(10)
    fig, ax = plt.subplots()
    ax.plot(random_data)
    ax.set_xlabel("X Axis Label")
    ax.set_ylabel("Y Axis Label")
    ax.set_title("Random Data Line Chart")
    st.pyplot(fig)

    # Display pie chart using Matplotlib
    st.subheader("Predicted Price vs. Input Data Pie Chart")
    pie_data = [prediction, sum(input_array[0])]
    labels = ["Predicted Price", "Input Data Total"]
    colors = ['blue', 'orange']
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)

    # Display heatmap using Seaborn
    st.subheader("Correlation Heatmap")
    correlation_matrix = np.random.rand(
        10, 10)  # Replace with your actual data
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


if __name__ == '__main__':
    main()
   
     
