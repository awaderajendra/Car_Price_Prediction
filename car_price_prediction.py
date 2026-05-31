import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("cardekho_dataset.csv")

print("Dataset:")
print(data.columns)

# Convert Brand Names into numbers
le = LabelEncoder()
data['brand'] = le.fit_transform(data['brand'])

# Input and Output
x = data[['mileage', 'vehicle_age', 'brand']]
y = data['selling_price']

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# sample data for prediction
sample_data = pd.DataFrame(
    [[18, 2, 1]],
    columns = ['mileage', 'vehicle_age', 'brand']
)

# Predict price
prediction = model.predict(sample_data)

print("\nPredicted Car Price:")
print(prediction)