# DalioHG.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pdfkit
from data_loader import load_price_data
from correlation_engine import (
    compute_correlation_matrix,
    get_correlation_stats,
    render_heatmap
)
from risk_metrics import (
    calculate_volatility,
    calculate_sharpe
)
from optimizer import optimize_weights
from simulator import monte_carlo_simulation, backtest_portfolio

st.set_page_config(page_title="Holy Grail Portfolio Analyzer", layout="wide")
st.title("Holy Grail Portfolio Analyzer")

header_cols = st.columns([1, 7, 1])
timeframe_selection = header_cols[0].selectbox(
    "Select Historical Period",
    options=["30 Days", "60 Days", "90 Days", "180 Days", "1 Year"],
    index=2
)
opt_mode = header_cols[1].radio("Weight Mode", ["Optimizer", "Manual Allocation"], horizontal=True)
run = header_cols[2].button("Run")

st.subheader("Enter Symbols")
ticker_cols = st.columns(7) + st.columns(7)
tickers = []
for i, col in enumerate(ticker_cols):
    ticker = col.text_input("Ticker", value="", label_visibility="collapsed", key=f"ticker_input_{i}", max_chars=10, placeholder=f"T{i+1}")
    if ticker:
        tickers.append(ticker.upper())

manual_weights = {}
if opt_mode == "Manual Allocation" and tickers:
    st.markdown("#### Manual Allocation (%)")
    alloc_cols = st.columns(7) + st.columns(7)
    for i, ticker in enumerate(tickers):
        col = alloc_cols[i]
        weight = col.number_input(f"{ticker}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"alloc_{ticker}")
        manual_weights[ticker] = weight / 100.0

timeframe_map = {
    "30 Days": 30,
    "60 Days": 60,
    "90 Days": 90,
    "180 Days": 180,
    "1 Year": 252
}
selected_days = timeframe_map[timeframe_selection]

report_html = ""
std_chart_base64 = ""
backtest_chart_base64 = ""
monte_carlo_chart_base64 = ""

def plot_std_vs_correlation(volatility, corr_matrix, weights):
    global std_chart_base64
    num_assets = len(volatility)
    avg_corr = corr_matrix.values[np.triu_indices(num_assets, 1)].mean()
    std_dev_lines = {}

    x_vals = list(range(1, 21))
    for corr_level in [0.0, 0.25, 0.5, 0.75, 1.0]:
        std_vals = [np.sqrt(corr_level + (1 - corr_level)/n) for n in x_vals]
        std_dev_lines[f"ρ = {corr_level:.2f}"] = std_vals

    cov_matrix = corr_matrix.copy()
    for i, ti in enumerate(volatility.index):
        for j, tj in enumerate(volatility.index):
            cov_matrix.iloc[i, j] = corr_matrix.iloc[i, j] * volatility[ti] * volatility[tj]
    weights_vec = np.array([weights.get(t, 0.0) for t in volatility.index])
    port_std = np.sqrt(weights_vec.T @ cov_matrix.values @ weights_vec)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, stds in std_dev_lines.items():
        ax.plot(x_vals, stds, label=label)

    ax.scatter([num_assets], [port_std], color='red', label='Your Portfolio', zorder=5)
    ax.set_title("Portfolio Standard Deviation vs. Number of Assets")
    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Std. (normalized)")
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    std_chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    st.pyplot(fig)

if run:
    if len(tickers) < 2:
        st.error("Please enter at least two valid tickers.")
    else:
        with st.spinner(f"Loading data and computing metrics for {selected_days} days..."):
            price_data = load_price_data(tickers, selected_days)
            st.write("Tickers Submitted:", tickers)
            st.write("Price Data Preview:")
            st.write(price_data)

            if price_data is None or price_data.shape[1] < 2:
                st.error("Not enough valid data to compute results.")
            else:
                corr = compute_correlation_matrix(price_data)

                if corr.empty or corr.shape[0] < 2:
                    st.error("Correlation matrix could not be computed — insufficient or invalid data.")
                else:
                    stats = get_correlation_stats(corr)
                    st.markdown("### Correlation Matrix")
                    metric_cols = st.columns(3)
                    avg_corr = stats["avg"]
                    if isinstance(avg_corr, pd.Series):
                        avg_corr = avg_corr.mean()
                    metric_cols[0].metric("Average Correlation", f"{avg_corr:.2f}")
                    metric_cols[1].metric(f"Max Correlation ({stats['max_pair'][0]} / {stats['max_pair'][1]})", f"{stats['max_val']:.2f}")
                    metric_cols[2].metric(f"Min Correlation ({stats['min_pair'][0]} / {stats['min_pair'][1]})", f"{stats['min_val']:.2f}")

                    fig = render_heatmap(corr)
                    st.pyplot(fig)

                    vol = calculate_volatility(price_data)
                    sharpe = calculate_sharpe(price_data)

                    weights = None
                    if opt_mode == "Optimizer":
                        st.markdown("###")
                        try:
                            weights = optimize_weights(price_data, mode="min_corr")
                            if weights.sum() < 0.01 or (weights == 0).all():
                                st.warning("Optimizer returned near-zero weights. Possibly unstable data or singular covariance matrix.")
                            else:
                                st.success("Optimization Completed")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")
                            weights = None
                    elif opt_mode == "Manual Allocation":
                        if abs(sum(manual_weights.values()) - 1.0) > 0.001:
                            st.error("Manual allocations must sum to 100%.")
                        else:
                            weights = pd.Series(manual_weights)

                    if weights is not None:
                        st.markdown("### Portfolio Metrics")
                        combined_df = pd.DataFrame({
                            "Ticker": tickers,
                            "Volatility (1YR)": [vol.get(t, float("nan")) for t in tickers],
                            "Sharpe Ratio": [sharpe.get(t, float("nan")) for t in tickers],
                            "Weight": [weights.get(t, 0.0) for t in tickers]
                        }).set_index("Ticker")

                        col_chart, col_table = st.columns([3, 2])
                        with col_table:
                            st.dataframe(combined_df.style.format({
                                "Volatility (1YR)": "{:.2%}",
                                "Sharpe Ratio": "{:.2f}",
                                "Weight": "{:.2%}"
                            }), use_container_width=False)
                        with col_chart:
                            plot_std_vs_correlation(vol, corr, weights)

                        st.markdown("### Simulations")
                        col_sim, col_bt = st.columns(2)
                        with col_bt:
                            st.markdown("**Historical Backtest**")
                            cumulative = backtest_portfolio(price_data, weights)
                            fig_bt, ax_bt = plt.subplots()
                            cumulative.plot(ax=ax_bt, title="")
                            buf_bt = BytesIO()
                            fig_bt.savefig(buf_bt, format="png", bbox_inches="tight")
                            buf_bt.seek(0)
                            backtest_chart_base64 = base64.b64encode(buf_bt.read()).decode("utf-8")
                            st.pyplot(fig_bt)

                        with col_sim:
                            st.markdown("**Monte Carlo Simulation**")
                            mc_df = monte_carlo_simulation(price_data, weights, n_simulations=250, n_days=selected_days)
                            fig_mc, ax_mc = plt.subplots()
                            mc_df.T.plot(ax=ax_mc, legend=False, title="")
                            buf_mc = BytesIO()
                            fig_mc.savefig(buf_mc, format="png", bbox_inches="tight")
                            buf_mc.seek(0)
                            monte_carlo_chart_base64 = base64.b64encode(buf_mc.read()).decode("utf-8")
                            st.pyplot(fig_mc)

                        report_html = f"""
                        <h1>Holy Grail Portfolio Report</h1>
                        <p>Tickers: {', '.join(tickers)}</p>
                        <p>Average Correlation: {stats['avg']:.2f}</p>
                        <p>Max Correlation: {stats['max_val']:.2f} ({stats['max_pair'][0]}/{stats['max_pair'][1]})</p>
                        <p>Min Correlation: {stats['min_val']:.2f} ({stats['min_pair'][0]}/{stats['min_pair'][1]})</p>
                        <h2>Portfolio Weights</h2>
                        {combined_df.to_html()}
                        """

                        if std_chart_base64:
                            report_html += f'''
                                <h2>Std Dev vs. Correlation Chart</h2>
                                <img src="data:image/png;base64,{std_chart_base64}" style="max-width:100%;height:auto;">
                            '''
                        if backtest_chart_base64:
                            report_html += f'''
                                <h2>Backtest Chart</h2>
                                <img src="data:image/png;base64,{backtest_chart_base64}" style="max-width:100%;height:auto;">
                            '''
                        if monte_carlo_chart_base64:
                            report_html += f'''
                                <h2>Monte Carlo Simulation Chart</h2>
                                <img src="data:image/png;base64,{monte_carlo_chart_base64}" style="max-width:100%;height:auto;">
                            '''

# --- PDF Export using pdfkit with explicit wkhtmltopdf path ---
if report_html:
    path_to_wkhtmltopdf = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

    pdf_bytes = BytesIO()
    pdf = pdfkit.from_string(report_html, False, configuration=config)
    pdf_bytes.write(pdf)
    pdf_bytes.seek(0)

    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="portfolio_report.pdf",
        mime="application/pdf"
    )