import pandas as pd
import streamlit as st
data = pd.read_csv('C:\\breast-cancer-wisconsin.data',names = [
    'Sample code number',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
])
def is_non_numrix(x):
    return not x.isnumeric()\

mask = data['Bare Nuclei'].apply(is_non_numrix)

data_non_numeric = data[mask]

data_numeric = data[~mask]

data_numeric['Bare Nuclei'] = data_numeric['Bare Nuclei'].astype('int64')

data_input = data_numeric.drop(columns = ['Sample code number','Class'])
data_output = data_numeric['Class']

data_output = data_output.replace({2:0,4:1})
from sklearn.metrics import  accuracy_score

from sklearn.model_selection import train_test_split
x,x_test, y,  y_test = train_test_split(data_input, data_output, test_size=1/3, random_state=2)

x_train,x_val, y_train,  y_val = train_test_split(x, y, test_size=1/3, random_state=2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3,random_state = 2)
model.fit(x_train, y_train)

y_pred_train =model.predict(x_train)
y_pred_val =model.predict(x_val)

max_depth_values = [1,2,3,4,5,6,7,8]
train_accuracy_values =[]
val_accuracy_values = []
for max_depth_val in max_depth_values:
    model = DecisionTreeClassifier(max_depth=max_depth_val,random_state = 2)
    model.fit(x_train, y_train)
    y_pred_train =model.predict(x_train)
    y_pred_val =model.predict(x_val)
    acc_train=accuracy_score(y_train,y_pred_train)
    acc_val=accuracy_score(y_val,y_pred_val)
    train_accuracy_values.append(acc_train)
    val_accuracy_values.append(acc_val)
    
final_model = DecisionTreeClassifier(max_depth=3,random_state = 0)
final_model.fit(x_train, y_train)

y_pred_test = final_model.predict(x_test)


    
st.title('Breast Cancer Project')
st.write('-----OUR TEAM-----')  
st.write('-Abdulrahman Elbanna')    
st.write('-Ahmed Yasser')    
st.write('-Omnia sameh')    

st.title('What is Breast Cancer?')
st.write('Breast cancer is one of the most common cancer diseases in worldwide; breast cancer happens in both men and women, generally more common in women.')



st.title('What does this model do?')
st.write('-This model predicts whether breast cancer is benign or malignant by entering some inputs')

st.title('We Use This Data')    
st.write(data_input)

import pickle

with open("C:/Users/compunil/saved-model.pickle",'rb') as f:
    model = pickle.load(f)
    
    

def app():
    # Add a title to the app
    st.title("My Predictive Model")
    
    # Add a description for the user
    st.write("Please enter values for the following features:")
    
    # Create input fields for each feature
    feature1 = st.number_input("Clump Thickness", min_value=0, max_value=10)
    feature2 = st.number_input("Uniformity of Cell Size", min_value=0, max_value=10)
    feature3 = st.number_input("Uniformity of Cell Shape", min_value=0, max_value=10)
    feature4 = st.number_input("Marginal Adhesion", min_value=0, max_value=100)
    feature5 = st.number_input("Single Epithelial Cell Size", min_value=0, max_value=10)
    feature6 = st.number_input("Bare Nuclei", min_value=0, max_value=10)
    feature7 = st.number_input("Bland Chromatin", min_value=0, max_value=10)
    feature8 = st.number_input("Normal Nucleoli", min_value=0, max_value=10)
    feature9 = st.number_input("Mitoses", min_value=0, max_value=10)
    
    # Create a dataframe from the inputs
    df = pd.DataFrame({
        'Clump Thickness': feature1,
        'Uniformity of Cell Size': feature2,
        'Uniformity of Cell Shape': feature3,
        'Marginal Adhesion': feature4,
        'Single Epithelial Cell Size': feature5,
        'Bare Nuclei': feature6,
        'Bland Chromatin': feature7,
        'Normal Nucleoli': feature8,
        'Mitoses': feature9,
        }, index=[0])
    
    # Make a prediction using the model and inputs
    prediction = model.predict(df)
    
    # Display the prediction to the user
    from PIL import Image
    st.write("\nThe predicted output is:", prediction[0])
    if prediction == 0:
        st.write('---------The disease is benign-------------')
        image = Image.open("D:\\2.jpeg")
        st.image(image, caption='Benign Tumor', width=400)
    else:
        st.write('---------The disease is malignant----------')
        image = Image.open("D:\\istockphoto-1062129234-612x612.jpg")
        st.image(image, caption='malignant Tumor', width=400)
        
app()

  

