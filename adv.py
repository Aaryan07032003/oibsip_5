import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import streamlit as st

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

file_path = 'Advertising.csv'
sales_data = load_data(file_path)

st.title("SALES PREDICTION ANALYSIS")

st.header("Basic Information")
st.write(sales_data.head())

st.header("Missing Values")
st.write(sales_data.isnull().sum())

st.header("Summary Statistics")
st.write(sales_data.describe())

st.header("Pairplot")
sns.pairplot(sales_data)
plt.title("Pairplot of Numerical Features")
st.pyplot()

st.header("Correlation Heatmap")
correlation_matrix = sales_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
st.pyplot()

# Preprocessing
imputer = SimpleImputer(strategy='mean')
sales_data_imputed = pd.DataFrame(imputer.fit_transform(sales_data), columns=sales_data.columns)

X = sales_data_imputed[['TV', 'Radio', 'Newspaper']]
y = sales_data_imputed['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.header("Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)
