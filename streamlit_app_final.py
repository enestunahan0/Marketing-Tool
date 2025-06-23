import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ml_modelleri import (
    load_future_costs_from_excel,
    order_tahmin,
    simulate_predictions_with_uncertainty,
    aggregate_predictions
)

# === Sayfa Yapılandırması ===
st.set_page_config(page_title="Order Forecasting App", layout="wide")
st.title("📦 Order Forecasting with Digital Marketing")
st.markdown("""
This application predicts the order quantities of the coming days with the digital marketing costs you have given.
Predictions are made with machine learning models trained based on historical data.
""")

# === Dosya Yükleme ===
uploaded_file = st.file_uploader("Upload future cost file (Excel)", type=["xlsx"])

if uploaded_file:
    future_costs = load_future_costs_from_excel(uploaded_file)
    st.success("File uploaded successfully.")
    st.dataframe(future_costs.head(10))

    # === Sidebar Ayarları ===
    with st.sidebar:
        st.header("⚙️ Forecast Settings")
        sim_mode = st.checkbox("I want to forecast with confidence interval")
        sim_count = st.slider("Number of simulations", min_value=1, max_value=300, value=50, step=10)
        confidence = st.selectbox("Confidence Interval (%)", options=[75, 80, 85, 90, 95], index=1)

    # === Tahmin Hesaplama ===
    with st.spinner("Calculating forecasts..."):
        if sim_mode:
            result_df = simulate_predictions_with_uncertainty(future_costs, n_simulations=sim_count)
            result_df = result_df.rename(columns={
                "mean_predicted_orders": "predicted_orders",
                "std_predicted_orders": "std",
                "lower_5th": "lower",
                "upper_95th": "upper"
            })
        else:
            base_result = order_tahmin(future_costs)
            result_df = base_result.copy()
            result_df['mean'] = result_df['predicted_orders']
            result_df['lower'] = result_df['mean']
            result_df['upper'] = result_df['mean']
            result_df['std'] = 0.0

    # === Sekmeler ===
    tab1, tab2, tab3 = st.tabs(["📑 Forecast Tables", "📈 Graphics", "📄 Data Preview"])

    # === TAB 1: Tahmin Tabloları ===
    with tab1:
        st.subheader("📅 Daily Forecasts")
        st.dataframe(result_df)
        st.download_button("📥 Download Daily Forecasts", result_df.to_csv(index=False), file_name="daily_forecasts.csv")

        # 3'er günlük, haftalık, aylık gruplama
        def show_aggregated(label, freq_code):
            df = result_df.copy()
            df['grup'] = df['date'].dt.to_period(freq_code).apply(lambda r: r.start_time)
            agg = df.groupby('grup')[['predicted_orders', 'lower', 'upper']].agg(['sum', 'mean'])
            st.markdown(f"#### {label} Forecasts")
            st.dataframe(agg)
            st.download_button(f"📥 {label} Download", agg.to_csv(), file_name=f"{label.lower()}_forecast.csv")

        show_aggregated("3 Days", '3D')
        show_aggregated("Weekly", 'W')
        show_aggregated("Monthly", 'M')

    # === TAB 2: Grafikler ===
    with tab2:
        st.subheader("📈 Daily Order Forecast Graph")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result_df['date'], y=result_df['predicted_orders'], mode='lines+markers', name='mean'))
        if sim_mode:
            fig.add_trace(go.Scatter(
                x=result_df['date'].tolist() + result_df['date'][::-1].tolist(),
                y=result_df['upper'].tolist() + result_df['lower'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f"%{confidence} Confidence Interval",
                hoverinfo="skip"
            ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Order", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        
            # === TAB 2: Grafikler === devamı
        st.subheader("📊 Actual and Forecast Order Comparison")

        uploaded_truth = st.file_uploader("Upload the actual sales file (Excel)", type=["xlsx"], key="truth_upload")
        if uploaded_truth:
            gerçek_df = pd.read_excel(uploaded_truth)
            gerçek_df['date'] = pd.to_datetime(gerçek_df['date'])
            karşılaştırma_df = pd.merge(result_df, gerçek_df, on='date', how='inner')

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=karşılaştırma_df['date'], y=karşılaştırma_df['mean'], mode='lines+markers', name='Forecast'))
            fig2.add_trace(go.Scatter(x=karşılaştırma_df['date'], y=karşılaştırma_df['actual_orders'], mode='lines+markers', name='Real Values'))

            fig2.update_layout(title="Actual vs Estimated Number of Orders", xaxis_title="Date", yaxis_title="Order Quantity", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📉 Error Metrics")

            from sklearn.metrics import mean_squared_error, mean_absolute_error

            mse = mean_squared_error(karşılaştırma_df["actual_orders"], karşılaştırma_df["mean"])
            rmse = mse ** 0.5
            mae = mean_absolute_error(karşılaştırma_df["actual_orders"], karşılaştırma_df["mean"])
            mape = (abs((karşılaştırma_df["actual_orders"] - karşılaştırma_df["mean"]) / karşılaştırma_df["actual_orders"])).mean() * 100

            st.write(f"**RMSE** (Root Mean Square Error): {rmse:.2f}")
            st.write(f"**MAE** (Mean Absolute Error): {mae:.2f}")
            st.write(f"**MAPE** (Mean Absolute Percentage Error): %{mape:.2f}")

            
    # === TAB 3: Veri Önizleme ===
    with tab3:
        st.subheader("📄 Loaded Cost Data")
        st.dataframe(future_costs)
        st.download_button("📥 Download Cost Data", future_costs.to_csv(index=False), file_name="cost_data.csv")
