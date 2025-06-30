import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
import io

st.set_page_config(page_title="CO₂ Forecasting with GRU", layout="centered")

st.title("📈 GRU Forecasting Emisi CO₂")
st.markdown("Upload file `.csv` yang memiliki kolom: **Year** dan **CO2**")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validasi kolom wajib
        if not {'Year', 'CO2'}.issubset(df.columns):
            st.error("❌ Kolom wajib tidak ditemukan. Harus ada kolom: 'Year' dan 'CO2'")
        else:
            df = df[['Year', 'CO2']].dropna()
            df['Year'] = df['Year'].astype(int)

            st.success("✅ File berhasil dimuat!")
            st.subheader("📊 Visualisasi Awal")
            st.line_chart(df.set_index('Year')['CO2'])

            # Forecasting
            st.subheader("🔮 Forecasting GRU")
            window = st.slider("Ukuran Window", 5, 20, 10)
            horizon = st.slider("Horizon Prediksi (tahun)", 1, 10, 5)
            epochs = st.slider("Epoch Training", 50, 500, 200, step=50)

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[['CO2']])
            X, y = [], []
            for i in range(len(scaled) - window):
                X.append(scaled[i:i+window])
                y.append(scaled[i+window])
            X, y = np.array(X), np.array(y)

            if len(X) < 10:
                st.warning("❗ Data tidak cukup untuk pelatihan model.")
            else:
                model = Sequential([
                    GRU(64, input_shape=(X.shape[1], X.shape[2])),
                    Dense(1)
                ])
                model.compile(loss='mse', optimizer='adam')
                with st.spinner("Melatih model..."):
                    model.fit(X, y, epochs=epochs, verbose=0)

                pred = model.predict(X)
                y_true = scaler.inverse_transform(y.reshape(-1, 1))
                y_pred = scaler.inverse_transform(pred)

                last_window = scaled[-window:]
                future_pred = []
                for _ in range(horizon):
                    input_seq = last_window.reshape(1, window, 1)
                    next_pred = model.predict(input_seq, verbose=0)
                    future_pred.append(next_pred[0, 0])
                    last_window = np.append(last_window[1:], next_pred, axis=0)
                future_actual = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1)).flatten()
                future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 1 + horizon)

                # Plot hasil prediksi
                st.subheader("📈 Grafik Hasil Prediksi")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['Year'][window:], y_true, label='Actual')
                ax.plot(df['Year'][window:], y_pred, label='Predicted')
                ax.plot(future_years, future_actual, 'o--', label='Forecast')
                ax.set_xlabel("Tahun")
                ax.set_ylabel("CO₂ Emissions")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                st.markdown("### 📌 Hasil Forecast:")
                for yr, val in zip(future_years, future_actual):
                    st.write(f"Tahun **{yr}**: {val:.2f} juta ton")

                # Anomaly Detection
                st.subheader("🚨 Deteksi Anomali")
                errors = np.abs(y_true - y_pred)
                mae_per_sample = np.mean(errors, axis=1)
                threshold = np.mean(mae_per_sample) + 2 * np.std(mae_per_sample)
                anomalies_years = df['Year'][window:][mae_per_sample > threshold]
                anomalies_errors = mae_per_sample[mae_per_sample > threshold]

                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(df['Year'][window:], mae_per_sample, label='MAE')
                ax2.axhline(threshold, color='red', linestyle='--', label='Threshold')
                ax2.scatter(anomalies_years, anomalies_errors, color='red', label='Anomaly')
                ax2.set_xlabel("Tahun")
                ax2.set_ylabel("MAE")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)

                if len(anomalies_years) > 0:
                    st.warning("Tahun terdeteksi anomali:")
                    st.write(anomalies_years.values)
                else:
                    st.info("Tidak ada anomali yang terdeteksi.")

    except Exception as e:
        st.error(f"❌ Gagal memproses file: {e}")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")

