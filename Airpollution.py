import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate

# Load data
data = pd.read_csv("D:\\cnn_air_quality_new.csv")

# Calculate AQI
def calculate_aqi(so2, no2, rspm_pm10, pm25):
    return so2 + no2 + rspm_pm10 + pm25

data['AQI'] = data.apply(lambda row: calculate_aqi(row['SO2'], row['NO2'], row['RSPM/PM10'], row['PM 2.5']), axis=1)

# Fill missing values
columns_to_fill = ['SO2', 'NO2', 'RSPM/PM10', 'PM 2.5']
for column in columns_to_fill:
    data[column] = data[column].fillna(data[column].mean())

# Define AQI category function
def classify_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif 51 <= aqi <= 100:
        return 'Moderate'
    elif 101 <= aqi <= 200:
        return 'Satisfactory'
    elif 201 <= aqi <= 300:
        return 'Poor'
    elif 301 <= aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Prepare data
X = data[['SO2', 'NO2', 'RSPM/PM10', 'PM 2.5']]
y = data['AQI']

# Reshape for CNN and split data
X_2d = np.array(X).reshape(-1, 2, 2, 1)
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# CNN model
cnn_input = Input(shape=(2, 2, 1))
cnn_model = Conv2D(32, (2, 2), activation='relu')(cnn_input)
cnn_model = Flatten()(cnn_model)
cnn_model = Dense(64, activation='relu')(cnn_model)

# ANN model
ann_input = Input(shape=(4,))
ann_model = Dense(64, activation='relu')(ann_input)
ann_model = Dense(32, activation='relu')(ann_model)

# Combine models
combined = concatenate([cnn_model, ann_model])
output = Dense(1, activation='linear')(combined)
model = Model(inputs=[cnn_input, ann_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train, X_train.reshape(-1, 4)], y_train, epochs=5, batch_size=32, validation_split=0.2)
y_pred = model.predict([X_test, X_test.reshape(-1, 4)])

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

# Classify test data into AQI categories
y_test_categories = y_test.apply(classify_aqi)

# Classify predicted data into AQI categories
y_pred_categories = pd.Series(y_pred.flatten()).apply(classify_aqi)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(data[['SO2', 'NO2', 'RSPM/PM10', 'PM 2.5', 'AQI']].corr(),
                      annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                      center=0, cbar_kws={'shrink': 0.8})

# Improve Title and Labels
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.show()

# Calculate residuals
residuals = y_test.values.flatten() - y_pred.flatten()
plt.scatter(y_test, residuals, alpha=0.5)
plt.title('Residuals Plot')
plt.xlabel('Actual AQI')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')  # Adding a horizontal line at y=0
plt.show()

sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.show()

# Compare actual vs. predicted AQI categories
category_comparison = pd.DataFrame({'Actual': y_test_categories, 'Predicted': y_pred_categories})
print(category_comparison)

# Plotting detailed pollution plot based on Type of Location using a violin plot
plt.figure(figsize=(12, 8))
# Reset the index to avoid duplicate labels
data_reset_index = data.reset_index(drop=True)
sns.swarmplot(x='Type of Location', y='AQI', data=data_reset_index)
plt.title('Distribution of Pollution based on Type of Location')
plt.xlabel('Type of Location')
plt.ylabel('AQI')
plt.show()

# Plotting pollution prediction plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.show()

# Reset the index to avoid duplicate labels
data_reset_index = data.reset_index(drop=True)

# Plot line graphs for each 'City/Town/Village/Area'
plt.figure(figsize=(12, 6))
sns.lineplot(x='City/Town/Village/Area', y='Health_rate', data=data_reset_index, marker='o', ci=None, hue='City/Town/Village/Area')
plt.xticks(rotation=45, ha='right')
plt.title('Healthrate vs City/Town/Village/Area')
plt.show()

# Plot line graphs for each 'Location of Monitoring Station'
plt.figure(figsize=(12, 6))
sns.lineplot(x='Location of Monitoring Station', y='Health_rate', data=data_reset_index, marker='o', ci=None, hue='Location of Monitoring Station')
plt.xticks(rotation=45, ha='right')
plt.title('Healthrate vs Location of Monitoring Station')
plt.show()
