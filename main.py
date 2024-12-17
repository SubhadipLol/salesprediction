import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = "sales_with_duplicate.csv"  # Replace with your file path
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Debug column names and data
print("Columns in the dataset:")
print(df.columns)

print("\nSample data:")
print(df.head())

# Data cleaning for 'No_of_reviews'
try:
    df['No_of_reviews'] = pd.to_numeric(
        df['No_of_reviews'].str.replace(',', '', regex=True).str.strip(), errors='coerce'
    ).fillna(0).astype(int)
except Exception as e:
    print(f"Error cleaning 'No_of_reviews' column: {e}")

# Rename columns safely with checks
columns_to_rename = {
    'product_price': 'Price',
    'rating_val': 'Ratings',
    'reviews': 'No_of_reviews',
    'product_name': 'Product_Name',
    'product_year': 'Year',
    'category': 'Product_Type',
    'product_description': 'Product_Description'
}

df.rename(columns=columns_to_rename, errors='ignore', inplace=True)

# Ensure 'Product_Type' is available in DataFrame
if 'Product_Type' not in df.columns:
    print("\nError: 'Product_Type' column is missing after renaming. Available columns are:")
    print(df.columns)
    exit()

# Feature Engineering for price categorization
def categorize_price(price):
    if price < 20000:
        return 'Economic'
    elif 20000 <= price <= 50000:
        return 'Mid-range'
    else:
        return 'Premium'

df['Price_Category'] = df['Price'].apply(categorize_price)

# Initialize dictionaries for models and predictions
all_models = {}
yearly_predictions = {}

# Train models for each year in the dataset
def train_models(data):
    global all_models, yearly_predictions
    years = data['year'].unique()

    for year in years:
        df_year = data[data['year'] == year]
        if df_year.empty:
            continue

        # Prepare features and labels
        features = ['Price', 'Ratings', 'No_of_reviews']
        # Convert to numeric, handling errors and non-breaking spaces
        for feature in features:
            df_year[feature] = pd.to_numeric(
                df_year[feature].astype(str).str.replace(',', '', regex=True).str.replace('\xa0', '', regex=True).str.strip(),
                errors='coerce'
            ).fillna(0)

        X = df_year[features].to_numpy()
        # Normalize features
        scaler_mean = np.mean(X, axis=0)
        scaler_std = np.std(X, axis=0)
        X_normalized = (X - scaler_mean) / scaler_std

        # Define the target variable 'y' here
        # Assuming 'Price_Category' is your target
        y = df_year['Price_Category']

        # Encode target labels
        unique_labels, y_numeric = np.unique(y, return_inverse=True)
        y_one_hot = pd.get_dummies(y_numeric).to_numpy()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_one_hot, test_size=0.2, random_state=42
        )

        # Define the model
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(y_one_hot.shape[1], activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

        # Save the trained model
        all_models[year] = model

        # Make predictions
        predictions = model.predict(X_normalized)
        df_year['Predicted_Category'] = [unique_labels[np.argmax(pred)] for pred in predictions]
        yearly_predictions[year] = df_year

# Train models using the cleaned dataset
train_models(df)

# Visualization function
def visualize_insights_by_year(year):
    """Visualize insights for a specific year."""
    if year not in yearly_predictions:
        print(f"No data available for the year {year}.")
        return

    df_year = yearly_predictions[year]

    # Plotting Price vs Ratings
    plt.figure(figsize=(10, 6))
    plt.scatter(df_year['Price'], df_year['Ratings'], c='blue', alpha=0.5, label='Price vs Ratings')
    plt.xlabel('Price')
    plt.ylabel('Ratings')
    plt.title(f'Price vs Ratings for Year {year}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Distribution of Predicted Categories
    category_counts = df_year['Predicted_Category'].value_counts()
    category_counts.plot(kind='bar', color='orange', figsize=(8, 5))
    plt.title(f'Predicted Category Distribution for Year {year}')
    plt.xlabel('Predicted Category')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.show()

    print(f"Insights for Year {year} Visualized Successfully.")

# Export analysis reports
def export_analysis_reports():
    """Export predictions and insights for all years into CSV files."""
    print("\nExporting reports...")
    for year, df_year in yearly_predictions.items():
        folder_path = 'csv_predictions/'
        # Make sure the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Construct the full file path
        # filename = f"predictions_analysis_{year}.csv"
        filename = os.path.join(folder_path, f"predictions_analysis_{year}.csv")   
        df_year.to_csv(filename, index=False, columns=[
            'Product_Name', 'Product_Description', 'Product_Type', 'Price', 'rating',
            'No_of_reviews', 'Predicted_Category'
        ])
        print(f"Saved predictions for year {year} to '{filename}'.")

    # Combine all yearly data for a comprehensive report
    full_report = pd.concat(yearly_predictions.values(), ignore_index=True)
    full_report.to_csv("comprehensive_analysis.csv", index=False, columns=[
        'Product_Name', 'Product_Description', 'Product_Type', 'Price', 'Ratings',
        'No_of_reviews', 'Predicted_Category'
    ])
    print("Saved comprehensive analysis to 'comprehensive_analysis.csv'.")

# Display predictions for each category
def display_predictions_by_category():
    """Display at least 10 predictions for each price category."""
    combined_df = pd.concat(yearly_predictions.values(), ignore_index=True)
    categories = ['Premium', 'Mid-range', 'Economic']

    for category in categories:
        print(f"\nTop 10 predictions for {category} products:")
        category_df = combined_df[combined_df['Predicted_Category'] == category].head(10)
        if not category_df.empty:
            print(category_df[['Product_Name', 'Product_Description', 'Product_Type',
                               'Price', 'Ratings', 'No_of_reviews', 'Predicted_Category']])
        else:
            print(f"No predictions available for {category} products.")

# Interactive menu system
def main_menu():
    while True:
        print("""
        1. Visualize Insights for a Specific Year
        2. View Top Product Predictions by Category
        3. Export Analysis Reports
        4. Display Predictions by Price Category
        5. Exit
        """)
        choice = input("Enter your choice: ")

        if choice == '1':
            try:
                year = int(input("Enter the year to visualize insights: "))
                visualize_insights_by_year(year)
            except ValueError:
                print("Invalid input! Please enter a valid year.")
        elif choice == '2':
            print("\nTop Product Predictions:")
            top_products = pd.concat(yearly_predictions.values()).sort_values(
                by=['Price', 'Ratings', 'No_of_reviews'],
                ascending=False
            ).head(10)
            print(top_products[['Product_Name', 'Product_Description', 'Product_Type',
                                'Price', 'Ratings', 'No_of_reviews', 'Predicted_Category']])
        elif choice == '3':
            export_analysis_reports()
        elif choice == '4':
            display_predictions_by_category()
        elif choice == '5':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice! Please select a valid option.")

# Start the main menu
main_menu()