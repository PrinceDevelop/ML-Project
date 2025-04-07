import numpy as np
import tkinter as tk
from tkinter import messagebox
import joblib

# Load trained models
RF_model = joblib.load("random_forest_model.pkl")
# LR_model = joblib.load("logistic_regression_model.pkl")

# Function to predict heart disease
def predict_disease():
    try:
        # Get values from user input
        age = int(age_entry.get())
        bp = int(bp_entry.get())
        chol = int(chol_entry.get())
        exercise = exercise_var.get()
        smoking = smoking_var.get()
        high_bp = high_bp_var.get()
        low_hdl = low_hdl_var.get()
        alcohol = alcohol_var.get()
        sleep = int(sleep_entry.get())

        # Prepare input array
        user_input = np.array([[age, bp, chol, exercise, smoking, high_bp, low_hdl, alcohol, sleep]])

        # Predictions from both models
        rf_prediction = RF_model.predict(user_input)[0]
        # lr_prediction = LR_model.predict(user_input)[0]

        # Determine result
        rf_result = "High Risk" if rf_prediction == 1 else "Low Risk"
        # lr_result = "High Risk" if lr_prediction == 1 else "Low Risk"

        # Show results
        messagebox.showinfo("Prediction", f"Random Forest: {rf_result}")
                            # \nLogistic Regression: {lr_result}")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# GUI Setup
root = tk.Tk()
root.title("Heart Disease Prediction")

# Labels and Entry Fields
tk.Label(root, text="Age:").grid(row=0, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1)

tk.Label(root, text="Blood Pressure:").grid(row=1, column=0)
bp_entry = tk.Entry(root)
bp_entry.grid(row=1, column=1)

tk.Label(root, text="Cholesterol Level:").grid(row=2, column=0)
chol_entry = tk.Entry(root)
chol_entry.grid(row=2, column=1)

tk.Label(root, text="Sleep Hours:").grid(row=3, column=0)
sleep_entry = tk.Entry(root)
sleep_entry.grid(row=3, column=1)

# Dropdowns for categorical variables
exercise_var = tk.IntVar()
tk.Label(root, text="Exercise Habits:").grid(row=4, column=0)
tk.OptionMenu(root, exercise_var, 0, 1, 2).grid(row=4, column=1)

smoking_var = tk.IntVar()
tk.Label(root, text="Smoking:").grid(row=5, column=0)
tk.OptionMenu(root, smoking_var, 0, 1).grid(row=5, column=1)

high_bp_var = tk.IntVar()
tk.Label(root, text="High Blood Pressure:").grid(row=6, column=0)
tk.OptionMenu(root, high_bp_var, 0, 1).grid(row=6, column=1)

low_hdl_var = tk.IntVar()
tk.Label(root, text="Low HDL Cholesterol:").grid(row=7, column=0)
tk.OptionMenu(root, low_hdl_var, 0, 1).grid(row=7, column=1)

alcohol_var = tk.IntVar()
tk.Label(root, text="Alcohol Consumption:").grid(row=8, column=0)
tk.OptionMenu(root, alcohol_var, 0, 1).grid(row=8, column=1)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict_disease)
predict_button.grid(row=9, column=0, columnspan=2)

# Run GUI
root.mainloop()
