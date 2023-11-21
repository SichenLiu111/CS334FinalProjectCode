import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'aia.us.csv'  # File path

data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# Simple Return (Close - Open)
df['SimpleReturn'] = (df['Close'] - df['Open'])/df['Open']

# Log Return Using Close price
# Log Return = ln(Close_t / Close_t-1) * 100
df['Log_Return%'] = np.log(df['Close'] / df['Close'].shift(1)) * 100

# Replace all empty spaces or null values with NaN
df = df.fillna(value=np.nan)
df = df.drop('OpenInt', axis=1)  # axis=1 specifies that you want to drop a column, not a row

# Calculate Moving Average for Close Price (e.g., over a 5-day window) #可以改
window_size = 5
df['Simple_Moving_Average (SMA)'] = df['Close'].rolling(window=window_size).mean()

# Label
df['label'] = (df['Log_Return%'] > 0).astype(int)


# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract Year, Month, Day of Month, and Day of Week
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day_of_Month'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.day_name()  # This gives the name of the day

# # Save the modified data to a new CSV file named 'processed.csv'
# processed_file_path = 'processed.csv'  # File path for the processed data
# df.to_csv(processed_file_path, index=False)

# Split Dataset randomly, 70% for training, and 30% for testing
X = df.drop('label', axis = 1)
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Save the train and test sets to files
x_train.to_csv('xTrain.csv', index=False)
x_test.to_csv('xTest.csv', index=False)
y_train.to_csv('yTrain.csv', index=False)
y_test.to_csv('yTest.csv', index=False)

