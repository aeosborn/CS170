import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load datasets
eth_price = pd.read_csv("ETH_USD_11_20_2020-1_19_2021_historical_prices (3).csv", sep=";")
transactions = pd.read_csv("transactions_aggregated.csv")
validators = pd.read_csv("validators_aggregated.csv")

# 2. Clean and preprocess ETH price data
eth_price['datetime'] = pd.to_datetime(eth_price['timestamp'], errors='coerce')
eth_price = eth_price[['datetime', 'close']].dropna()
eth_price['close'] = pd.to_numeric(eth_price['close'], errors='coerce')
eth_price = eth_price.dropna()

# 3. Clean and preprocess transaction data
transactions['datetime'] = pd.to_datetime(transactions['datetime'], errors='coerce')
transactions['valueETH'] = pd.to_numeric(transactions['valueETH'], errors='coerce')
transactions['gasUsed'] = pd.to_numeric(transactions['gasUsed'], errors='coerce')
transactions = transactions.dropna(subset=['datetime', 'valueETH', 'gasUsed'])

# Aggregate transactions by day
tx_daily = transactions.groupby(transactions['datetime'].dt.date).agg({
    'valueETH': 'sum',
    'gasUsed': 'mean'
}).reset_index().rename(columns={'datetime': 'date'})

# 4. Clean and preprocess validator data
validators['datetime'] = pd.to_datetime(validators['datetime'], errors='coerce')
validators['valueETH'] = pd.to_numeric(validators['valueETH'], errors='coerce')
validators['gasUsed'] = pd.to_numeric(validators['gasUsed'], errors='coerce')
validators = validators.dropna(subset=['datetime', 'valueETH', 'gasUsed'])

# Aggregate validator data by day
validators_daily = validators.groupby(validators['datetime'].dt.date).agg({
    'valueETH': 'sum',
    'gasUsed': 'mean'
}).reset_index().rename(columns={'datetime': 'date'})
validators_daily.columns = ['date', 'validator_valueETH', 'validator_gasUsed']

# 5. Merge all datasets
eth_price['date'] = eth_price['datetime'].dt.date
merged = eth_price.merge(tx_daily, on='date', how='left')
merged = merged.merge(validators_daily, on='date', how='left')

# Fill any remaining missing values with 0
merged.fillna(0, inplace=True)

# 6. Label ETH price into Low, Medium, High using quantiles
price_quantiles = merged['close'].quantile([0.33, 0.66])
def classify_price(price):
    if price <= price_quantiles[0.33]:
        return 'Low'
    elif price <= price_quantiles[0.66]:
        return 'Medium'
    else:
        return 'High'
merged['price_label'] = merged['close'].apply(classify_price)

# 7. Add lag features (important temporal context)
merged['lag_close_1'] = merged['close'].shift(1).fillna(method='bfill')
merged['lag_close_2'] = merged['close'].shift(2).fillna(method='bfill')

# 8. Prepare features and labels
features = merged[['valueETH', 'gasUsed', 'validator_valueETH', 'validator_gasUsed', 'lag_close_1', 'lag_close_2']]
labels = merged['price_label']

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# 10. Train optimized Random Forest
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# 11. Evaluate model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 12. Plot confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
