# Height & Age Prediction System

## About the Project
This project uses Polynomial Regression, a Machine Learning technique, to analyze the relationship between age and height and predict height based on age.

## Features
- Load and display a CSV dataset
- Explore dataset statistics
- Visualize Age vs Height using Polynomial Regression
- Evaluate model performance using Train R², Test R², and Mean Absolute Error (MAE)
- Compare actual and predicted height
- Predict height for a selected age

## Technologies Used
- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn

## How to Run the Project

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install streamlit pandas matplotlib scikit-learn
   ```

3. Make sure `app.py` and `height_age.csv` are in the same folder.
4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Model Evaluation
The application displays Train R² Score, Test R² Score, and Mean Absolute Error (MAE) to help evaluate the model.

## Future Improvements
- Improve the dashboard design
- Add more model evaluation metrics
- Deploy the app online

## License
This project is available for educational and learning purposes.
