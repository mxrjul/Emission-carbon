import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pycountry
import pycountry_convert as pc
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("co2-land-use.csv")
df.columns = ['Entity', 'Code', 'Year', 'CO2']
df_indo = df[df['Entity'] == 'Indonesia'][['Year', 'CO2']].reset_index(drop=True)

def prepare_multistep_data(df, window_size, forecast_horizon):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['CO2']])
    X, y = [], []
    for i in range(len(scaled) - window_size - forecast_horizon + 1):
        X.append(scaled[i:i + window_size])
        y.append(scaled[i + window_size:i + window_size + forecast_horizon].flatten())
    return np.array(X), np.array(y), scaler

window = 10
horizon = 5
X_multi, y_multi, scaler = prepare_multistep_data(df_indo, window, horizon)

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, shuffle=False)

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_multi.shape[1], X_multi.shape[2])),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(horizon)
])
model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stop], verbose=1)

pred_test = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.flatten().reshape(-1, 1)).reshape(y_test.shape)
pred_test_inv = scaler.inverse_transform(pred_test.flatten().reshape(-1, 1)).reshape(pred_test.shape)
mse = mean_squared_error(y_test_inv, pred_test_inv)
print(f"Test MSE: {mse:.4f}")

def forecast_with_preprocessing(df, window=10, epochs=100, horizon=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['CO2']])
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        GRU(64),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, verbose=0)

    pred = model.predict(X)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred = scaler.inverse_transform(pred)

    last_window = scaled[-window:]
    forecast = []
    for _ in range(horizon):
        input_seq = last_window.reshape(1, window, 1)
        next_pred = model.predict(input_seq, verbose=0)
        forecast.append(next_pred[0, 0])
        last_window = np.append(last_window[1:], next_pred, axis=0)
    future_pred = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, label='Actual')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted')
    plt.plot(range(len(y_true), len(y_true) + horizon), future_pred, label='Forecast', marker='o', linestyle='-.')
    plt.title("Forecasting CO₂ Emissions")
    plt.xlabel("Timestep")
    plt.ylabel("CO₂ Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"MSE (Train): {mean_squared_error(y_true, y_pred):.2f}")
    for i, val in enumerate(future_pred):
        print(f"Tahun ke-{i+1}: {val:.2f}")

# Contoh penggunaan forecast
df_sample = pd.DataFrame({
    'Year': np.arange(1990, 2023),
    'CO2': np.random.uniform(5, 10, size=33)
})
forecast_with_preprocessing(df_sample, window=10, epochs=500, horizon=5)

# Mapping kode negara ke benua
def get_continent(code):
    try:
        country_alpha2 = pycountry.countries.get(alpha_3=code).alpha_2
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except:
        return None

codes = df[['Entity', 'Code']].drop_duplicates().reset_index(drop=True)
codes['Region'] = codes['Code'].apply(get_continent)
codes = codes.dropna(subset=['Region'])
df_region = df.merge(codes, on="Code", how="left").dropna(subset=["Region"])

def forecast_region_with_future(region_name, window=10, epochs=100, horizon=5, plot_start_year=2020):
    region_df = df_region[df_region['Region'] == region_name]
    agg = region_df.groupby('Year')['CO2'].sum().reset_index()

    if len(agg) < window + horizon:
        print(f"Tidak cukup data untuk wilayah {region_name}")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(agg[['CO2']])
    X, y = [], []
    for i in range(len(scaled) - window - horizon + 1):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window:i+window+horizon].flatten())
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return

    model = Sequential([
        GRU(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(horizon)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, verbose=0)

    last_window = scaled[-window:].reshape(1, window, 1)
    predicted_future = model.predict(last_window)
    predicted_future = scaler.inverse_transform(predicted_future.reshape(-1, 1)).flatten()

    future_years = np.arange(agg['Year'].iloc[-1] + 1, agg['Year'].iloc[-1] + 1 + horizon)
    plt.figure(figsize=(12, 6))
    plot_data = agg[agg['Year'] >= plot_start_year]
    plt.plot(plot_data['Year'], plot_data['CO2'], label='Actual')
    last_actual = plot_data['CO2'].iloc[-1]
    future_plot_years = np.concatenate([[plot_data['Year'].iloc[-1]], future_years])
    future_plot_values = np.concatenate([[last_actual], predicted_future])
    plt.plot(future_plot_years, future_plot_values, label='Forecast', linestyle='--')
    plt.title(f"Forecasting CO₂ - {region_name}")
    plt.xlabel("Year")
    plt.ylabel("CO₂ Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()

    pred_train = model.predict(X)
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    y_pred = scaler.inverse_transform(pred_train.reshape(-1, 1)).reshape(pred_train.shape)
    mse = mean_squared_error(y_true, y_pred)
    print(f"✅ {region_name} | MSE: {mse:.2f}")
    for yr, co2 in zip(future_years, predicted_future):
        print(f"Tahun {yr}: {co2:.2f} juta ton")

for reg in df_region['Region'].unique():
    if reg == 'World':
        continue
    forecast_region_with_future(reg, window=10, epochs=100, horizon=5)

# Deteksi Anomali
model = Sequential([
    GRU(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=100, verbose=0)

pred = model.predict(X)
y_true_inv = scaler.inverse_transform(y.flatten().reshape(-1, 1)).reshape(y.shape)
y_pred_inv = scaler.inverse_transform(pred.flatten().reshape(-1, 1)).reshape(pred.shape)

errors = np.abs(y_true_inv - y_pred_inv)
mae_per_sample = np.mean(errors, axis=1)
threshold = np.mean(mae_per_sample) + 2 * np.std(mae_per_sample)

prediction_years = df_indo['Year'].values[window:window + len(mae_per_sample)]
anomalies_years = prediction_years[mae_per_sample > threshold]
anomalies_errors = mae_per_sample[mae_per_sample > threshold]

plt.figure(figsize=(12, 4))
plt.plot(prediction_years, mae_per_sample, label='MAE')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.scatter(anomalies_years, anomalies_errors, color='red', label='Anomalies')
plt.title("Anomaly Detection - MAE per Window")
plt.xlabel("Year")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

print("Tahun Anomali:", anomalies_years)

