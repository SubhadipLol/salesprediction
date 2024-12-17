import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import io
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
import os
import base64
import io
from flask import jsonify, send_from_directory
from flask import send_file
# import zipfile
from flask import Flask, render_template, jsonify, request, send_file
import chardet

app = Flask(__name__)

# Declare df as a global variable
df = pd.DataFrame()
all_models = {}
yearly_predictions = {}

@app.route("/")
def home():
    # Serve the main page (HTML)
    return render_template("start.html")

@app.route('/choosemodel')
def choose():
    return render_template("choosemodel.html")
    
@app.route("/second")
def second():
    # Serve another HTML page
    return render_template("home.html")

@app.route("/read_csv")
def read_csv():
    try:
        file_path = 'sales_with_duplicate.csv'  # Ensure this is the correct path to your CSV file
        global df  # Indicate that we're modifying the global variable
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Load CSV into global df
        
        return jsonify(message="CSV File Read Successfully")
    except Exception as e:
        return jsonify(message=f"Error: {e}")


@app.route('/cleaing')
def clean():
    try:
        if 'No_of_reviews' in df.columns:
            df['No_of_reviews'] = pd.to_numeric(
                df['No_of_reviews'].str.replace(',', '', regex=True).str.strip(), errors='coerce'
            ).fillna(0).astype(int)

        columns_to_rename = {
            'product_price': 'Price',
            'rating_val': 'Ratings',
            'reviews': 'No_of_reviews',
            'product_name': 'Product_Name',
            'product_year': 'Year',
            'category': 'Product_Type',
            'product_description': 'Product_Description'
        }

        # Renaming columns in the DataFrame
        df.rename(columns=columns_to_rename, errors='ignore', inplace=True)

        # Ensure 'Product_Type' is available in DataFrame
        if 'Product_Type' not in df.columns:
            print("\nError: 'Product_Type' column is missing after renaming. Available columns are:")
            print(df.columns)
            return jsonify(message="Error: 'Product_Type' column is missing.")

        # Feature Engineering for price categorization
        def categorize_price(price):
            if price < 20000:
                return 'Economic'
            elif 20000 <= price <= 50000:
                return 'Mid-range'
            else:
                return 'Premium'

        # Apply price categorization
        df['Price_Category'] = df['Price'].apply(categorize_price)

        return jsonify(message="Data cleaned Successfully")

    except Exception as e:
        return jsonify(message=f"Error: {e}")


@app.route('/training')
def training():
    try:
        def train_models(data):
            global all_models, yearly_predictions, df
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

        print(f"Dataset size: {df.shape}")
        train_models(df)

        return jsonify(message="Data training complete")
    except Exception as e:
        return jsonify(message=f"Error:{e}")
    
    # Train models using the cleaned dataset


@app.route('/menu')
def menu():
    return render_template("menu.html")


@app.route('/visuals')
def visualspage():
    return render_template("visuals.html")

@app.route('/visualize', methods=['POST'])
def visualize_show():
    try:
        data = request.get_json()
        year = int(data.get('year', 0))  # Convert the year to an integer
        
        df_year = yearly_predictions.get(year)
        if df_year is None:
            return jsonify(message=f"No data available for the year {year}.")

        # Create a directory for saving the images
        image_dir = os.path.join(app.root_path, 'static', 'images')  # Adjust path as needed
        os.makedirs(image_dir, exist_ok=True)

        # Save the first plot (Price vs Ratings) in the images directory
        img1_path = os.path.join(image_dir, f'price_vs_ratings_{year}.png')
        plt.figure(figsize=(10, 6))
        plt.scatter(df_year['Price'], df_year['Ratings'], c='blue', alpha=0.5, label='Price vs Ratings')
        plt.xlabel('Price')
        plt.ylabel('Ratings')
        plt.title(f'Price vs Ratings for Year {year}')
        plt.legend()
        plt.grid(True)
        plt.savefig(img1_path)  # Save the plot into a file
        plt.close()  # Close the plot to avoid overlapping with the second plot

        # Save the second plot (Predicted Category Distribution) in the images directory
        img2_path = os.path.join(image_dir, f'predicted_category_distribution_{year}.png')
        category_counts = df_year['Predicted_Category'].value_counts()
        category_counts.plot(kind='bar', color='orange', figsize=(8, 5))
        plt.title(f'Predicted Category Distribution for Year {year}')
        plt.xlabel('Predicted Category')
        plt.ylabel('Count')
        plt.grid(axis='y')
        plt.savefig(img2_path)  # Save the second plot into a file
        plt.close()  # Close the plot

        # Return the paths of the saved images to the client
        return jsonify({
            'image_1': f'/static/images/{os.path.basename(img1_path)}',
            'image_2': f'/static/images/{os.path.basename(img2_path)}'
        })

    except Exception as e:
        return jsonify(message=f"Error: {e}")



@app.route('/price_category')
def price():
    return render_template('disppred.html')

@app.route('/display_pred')
def showtable():
    combined_df = pd.concat(yearly_predictions.values(), ignore_index=True)
    categories = ['Premium', 'Mid-range', 'Economic']
    category_df_list = []

    for category in categories:
        category_df = combined_df[combined_df['Predicted_Category'] == category].head(10)
        if not category_df.empty:
            category_df_list.append(category_df)

    all_category_df = pd.concat(category_df_list, ignore_index=True)
    table_html = all_category_df.to_html(classes='data', header="true", index=False)

    return table_html


@app.route('/top_prediction')
def topproducts ():
    return render_template("toppred.html")

@app.route('/top_predictions')
def display_prediction():
    top_products = pd.concat(yearly_predictions.values()).sort_values(
                by=['Price', 'Ratings', 'No_of_reviews'],
                ascending=False
            ).head(10)
    table_html2 = top_products.to_html(classes='data', header="true", index=False)
    return table_html2

@app.route('/export')
def export():
    folder_path = 'csv_predictions/'
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate CSV files for each year
    for year, df_year in yearly_predictions.items():
        filename = os.path.join(folder_path, f"predictions_analysis_{year}.csv")
        df_year.to_csv(filename, index=False, columns=[
            'Product_Name', 'Product_Description', 'Product_Type', 'Price', 'Ratings',
            'No_of_reviews', 'Predicted_Category'
        ])

    # Create a comprehensive report
    full_report = pd.concat(yearly_predictions.values(), ignore_index=True)
    full_report.to_csv(os.path.join(folder_path, "comprehensive_analysis.csv"), index=False, columns=[
        'Product_Name', 'Product_Description', 'Product_Type', 'Price', 'Ratings',
        'No_of_reviews', 'Predicted_Category'
    ])

    # Provide a message to the user via the rendered template
    return render_template("export.html")



@app.route('/model_two')
def modeltwo():
    return render_template('input.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # Ensure the file is in the request
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        app.logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not file.filename.endswith('.csv'):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({"error": "File is not a CSV"}), 400

    # Read the first 250 bytes to detect encoding
    raw_data = file.read(250)
    result = chardet.detect(raw_data)
    encoding = result['encoding']

    # Reset file pointer after reading the initial data
    file.seek(0)

    app.logger.info(f"Received file: {file.filename}, attempting to process CSV")

    try:
        # Read the CSV into a DataFrame with the detected encoding
        dataframe = pd.read_csv(file, encoding=encoding)
        # Log the first few rows of the dataframe for debugging
        app.logger.info(f"CSV file processed successfully. Columns: {dataframe.columns.tolist()}")
        return jsonify({"message": "File processed successfully", "columns": dataframe.columns.tolist()}), 200
    except Exception as e:
        app.logger.error(f"Error processing CSV file: {e}")
        return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
