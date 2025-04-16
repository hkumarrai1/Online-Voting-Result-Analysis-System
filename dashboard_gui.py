import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog  # Import filedialog for file selection
from PIL import Image, ImageTk
import os
import ttkbootstrap as tb
import joblib
import pandas as pd  # Import pandas to load the cleaned dataset

# Function to display charts
def show_chart(chart_path):
    if os.path.exists(chart_path):
        chart_window = tk.Toplevel()
        chart_window.title("Chart Viewer")
        img = Image.open(chart_path)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)  # Updated resizing method
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(chart_window, image=photo)
        label.image = photo
        label.pack()
    else:
        messagebox.showerror("Error", f"Chart not found: {chart_path}")

# Functions for each analysis
def top_parties_analysis():
    show_chart("charts/top_parties_wins.png")

def turnout_trend_analysis():
    show_chart("charts/turnout_trend.png")

def margin_distribution_analysis():
    show_chart("charts/margin_distribution.png")

def statewise_elections_analysis():
    show_chart("charts/statewise_elections.png")

# Load the trained model and label encoders
model = joblib.load("party_prediction_model.pkl")
state_encoder = joblib.load("state_label_encoder.pkl")  # Load the saved LabelEncoder for states
party_encoder = joblib.load("party_label_encoder.pkl")  # Load the saved LabelEncoder for parties

# Load cleaned dataset to fetch unique states
df = pd.read_csv("cleaned_loksabha_data.csv")
unique_states = sorted(df['state'].unique())  # Get unique state names

# Prediction functionality
def predict_party():
    try:
        # Get user inputs
        state = state_var.get()
        year = int(year_var.get())
        turnout = float(turnout_var.get())
        margin = float(margin_var.get())

        # Encode the state input
        encoded_state = state_encoder.transform([state])[0]

        # Prepare input for the model
        input_data = [[encoded_state, year, turnout, margin]]

        # Ensure input matches the training data structure
        training_columns = list(pd.read_csv("cleaned_loksabha_data.csv").drop(columns=['party']).columns)
        input_df = pd.DataFrame(input_data, columns=['state', 'year', 'Turnout', 'margin'])
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0

        # Reorder columns to match the training data
        input_df = input_df[training_columns]

        # Make prediction
        prediction = model.predict(input_df)

        # Decode the predicted party number to the party name
        predicted_party = party_encoder.inverse_transform(prediction)[0]

        # Display the predicted party
        messagebox.showinfo("Prediction Result", f"The predicted winning party is: {predicted_party}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or prediction failed: {e}")

# Function to update the dataset
def update_dataset():
    try:
        # Prompt user to select a new dataset file
        new_file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV Files", "*.csv")]
        )
        if new_file_path:
            # Replace the existing dataset with the new one
            os.replace(new_file_path, "cleaned_loksabha_data.csv")
            messagebox.showinfo("Success", "Dataset updated successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update dataset: {e}")

# Function to re-train the model
def retrain_model():
    try:
        # Run the model training script
        os.system("python 4_model_party_prediction.py")
        messagebox.showinfo("Success", "Model re-trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to re-train the model: {e}")

# Main GUI
root = tb.Window(themename="superhero")
root.title("Online Voting Result Analysis System")
root.geometry("800x600")

# Title Label
title_label = ttk.Label(root, text="Online Voting Result Analysis System", font=("Arial", 20, "bold"))
title_label.pack(pady=20)

# Tabbed Interface
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# EDA Tab
eda_tab = ttk.Frame(notebook)
notebook.add(eda_tab, text="EDA Analysis")

# EDA Buttons
btn_top_parties = ttk.Button(eda_tab, text="Top 10 Parties by Wins", command=top_parties_analysis)
btn_top_parties.pack(pady=10)

btn_turnout_trend = ttk.Button(eda_tab, text="Voter Turnout Trend", command=turnout_trend_analysis)
btn_turnout_trend.pack(pady=10)

btn_margin_distribution = ttk.Button(eda_tab, text="Victory Margin Distribution", command=margin_distribution_analysis)
btn_margin_distribution.pack(pady=10)

btn_statewise_elections = ttk.Button(eda_tab, text="State-wise Election Participation", command=statewise_elections_analysis)
btn_statewise_elections.pack(pady=10)

# Prediction Tab
prediction_tab = ttk.Frame(notebook)
notebook.add(prediction_tab, text="Prediction")

# Prediction Placeholder
predict_label = ttk.Label(prediction_tab, text="Enter details for prediction", font=("Arial", 14))
predict_label.pack(pady=10)

# Dropdown options
states = unique_states  # Use unique states from the dataset
years = [str(year) for year in range(1962, 2024, 5)]  # Election years
turnout_options = [str(i) for i in range(40, 101, 5)]  # Turnout percentages (40% to 100%)
margin_options = [str(i) for i in range(1, 51, 5)]  # Margin of victory (1% to 50%)

# State Dropdown
state_label = ttk.Label(prediction_tab, text="State:")
state_label.pack(pady=5)
state_var = tk.StringVar()
state_dropdown = ttk.Combobox(prediction_tab, textvariable=state_var, values=states, state="readonly")
state_dropdown.pack(pady=5)

# Year Dropdown
year_label = ttk.Label(prediction_tab, text="Year:")
year_label.pack(pady=5)
year_var = tk.StringVar()
year_dropdown = ttk.Combobox(prediction_tab, textvariable=year_var, values=years, state="readonly")
year_dropdown.pack(pady=5)

# Turnout Dropdown
turnout_label = ttk.Label(prediction_tab, text="Turnout (%):")
turnout_label.pack(pady=5)
turnout_var = tk.StringVar()
turnout_dropdown = ttk.Combobox(prediction_tab, textvariable=turnout_var, values=turnout_options, state="readonly")
turnout_dropdown.pack(pady=5)

# Margin Dropdown
margin_label = ttk.Label(prediction_tab, text="Margin of Victory:")
margin_label.pack(pady=5)
margin_var = tk.StringVar()
margin_dropdown = ttk.Combobox(prediction_tab, textvariable=margin_var, values=margin_options, state="readonly")
margin_dropdown.pack(pady=5)

# Predict Button
predict_button = ttk.Button(prediction_tab, text="Predict Winning Party", command=predict_party)
predict_button.pack(pady=10)

# Settings Tab
settings_tab = ttk.Frame(notebook)
notebook.add(settings_tab, text="Settings")

# Settings Placeholder
settings_label = ttk.Label(settings_tab, text="Settings and Customization", font=("Arial", 14))
settings_label.pack(pady=20)

# Update Dataset Button
update_dataset_button = ttk.Button(settings_tab, text="Update Dataset", command=update_dataset)
update_dataset_button.pack(pady=10)

# Re-train Model Button
retrain_model_button = ttk.Button(settings_tab, text="Re-train Model", command=retrain_model)
retrain_model_button.pack(pady=10)

# Run the application
root.mainloop()
