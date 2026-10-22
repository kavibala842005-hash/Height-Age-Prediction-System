
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Height & Age Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Height & Age Prediction System")
st.subheader("Machine Learning Based Height Prediction Using Polynomial Regression")
st.write(
    "Analyze height and age data, visualize polynomial regression, "
    "evaluate model performance, and predict height using age."
)

# Load dataset
df = pd.read_csv("height_age.csv")

st.write("### Dataset")
st.dataframe(df)

st.write("### Dataset Statistics")
st.write(df.describe())

st.write("### Select Dataset Columns")

age_col = st.selectbox("Select Age Column", df.columns)
height_col = st.selectbox("Select Height Column", df.columns)

if age_col != height_col:
    data = df[[age_col, height_col]].dropna()

    X = data[[age_col]]
    y = data[height_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = st.slider("Polynomial Degree", 2, 5, 2)

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    test_predictions = model.predict(X_test_poly)
    comparison = pd.DataFrame({
    "Actual Height": y_test.values,
    "Predicted Height": test_predictions,
    "Prediction Error": abs(y_test.values - test_predictions)})

    st.write("### Actual vs Predicted Height")
    st.dataframe(comparison.round(2))
    mae = mean_absolute_error(y_test, test_predictions)

    train_predictions = model.predict(X_train_poly)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Train R² Score",
            f"{r2_score(y_train, train_predictions):.3f}"
        )

    with col2:
        st.metric(
            "Test R² Score",
            f"{r2_score(y_test, test_predictions):.3f}"
        )

    with col3:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual Height")

    sorted_data = data.sort_values(age_col)
    sorted_X = sorted_data[[age_col]]
    sorted_poly = poly.transform(sorted_X)
    sorted_predictions = model.predict(sorted_poly)

    ax.plot(
        sorted_X[age_col],
        sorted_predictions,
        color="red",
        label="Polynomial Regression"
    ) 
    ax.set_xlabel(age_col)
    ax.set_ylabel(height_col)
    ax.legend()
    st.write("## 📈 Height vs Age Analysis")
    st.pyplot(fig)

    st.write("###Height Prediction")

    age_value = st.number_input(
        "Enter Age",
        min_value=float(data[age_col].min()),
        max_value=float(data[age_col].max()),
        value=float(data[age_col].min())
    )

    if st.button("Predict Height"):
        input_poly = poly.transform(pd.DataFrame({age_col: [age_value]}))
        predicted_height = model.predict(input_poly)[0]
        st.success(f"Predicted Height: {predicted_height:.2f}")
else:
    st.warning("Please select different columns for Age and Height.")
