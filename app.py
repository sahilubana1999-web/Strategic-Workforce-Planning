# app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="SWP — Manpower Demand Forecasting")

import pandas as pd
import numpy as np
from datetime import datetime
import io

# Forecasting libs
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as e:
    ARIMA = None

from sklearn.linear_model import LinearRegression
import plotly.express as px

st.title("Strategic Workforce Planning — Manpower Demand Forecasting")
st.markdown(
    """
This app forecasts demand (output) and converts it to required manpower using productivity ratios.
Pick a forecasting model (Prophet recommended), upload historical data or use the sample dataset,
then review workforce gaps and cost implications.
"""
)

# -----------------------
# Sidebar: Inputs & config
# -----------------------
st.sidebar.header("Configuration")

if 'prod_method_radio' not in st.session_state:
    st.session_state['prod_method_radio'] = "Use recent observed (avg)"

model_choice = st.sidebar.selectbox("Forecasting model", options=["Prophet (recommended)", "ARIMA", "Linear Regression"])
horizon_months = st.sidebar.slider("Forecast horizon (months)", 1, 36, 6)
st.sidebar.radio(
    "Productivity per employee (monthly)",
    options=["Use recent observed (avg)", "Manual input"],
    key="prod_method_radio"
)

is_manual_input = (st.session_state['prod_method_radio'] == "Manual input")
disabled_state = not is_manual_input

recent_months = st.sidebar.slider("If using observed: months to average", 3, 24, 6, disabled=is_manual_input)

manual_prod_value = st.sidebar.number_input(
    "Manual productivity (output per employee per month)",
    value=20.0, 
    min_value=0.1, 
    max_value=100000.0,
    disabled=disabled_state
)

st.sidebar.markdown("### Financial assumptions (editable)")
avg_annual_salary = st.sidebar.number_input("Average annual salary (currency units)", 500000.0, step=10000.0)
benefits_pct = st.sidebar.slider("Benefits as % of salary", 0.0, 1.0, 0.25)
gross_margin = st.sidebar.slider("Estimated gross margin (for lost-profit calc)", 0.0, 1.0, 0.30)
lost_sales_factor = st.sidebar.slider("Lost-sales factor per missing employee (0-1)", 0.0, 1.0, 0.5)

with st.sidebar.expander("ℹ️ Help / Glossary of Terms"):
    st.markdown("""
    ### Key Terms Explained
    - **Configuration**: Settings that control how the forecast is generated.
    - **Forecast Horizon (months)**: Number of future months to predict manpower needs.
    - **Productivity per Employee (monthly)**: Average output one employee delivers per month.
    - **If using observed: Months to average**: Number of past months used to calculate average productivity.
    - **Financial Assumptions (editable)**: Salary, benefits %, margin, and other cost-related inputs.
    - **Benefits as % of Salary**: Extra costs like PF, insurance, etc., as a percentage of salary.
    - **Estimated Gross Margin (for lost-profit calc)**: Profit margin used to estimate lost profit when understaffed.
    - **Lost-sales Factor per Missing Employee (0–1)**: Proportion of sales lost when one employee is missing.
    - **Manual Productivity (output per employee per month)**: Manually override calculated productivity with a fixed value.
    - **Manual Productivity set to**: Example → 20.05 output/employee/month, used in manpower demand formula.
    - **Available Workforce (current headcount)**: Current employee headcount to compare against forecasted demand.
    - **Latest Dataset Headcount**: Last recorded headcount in uploaded dataset (baseline workforce).
    """)

# -----------------------
# Data upload / demo data
# -----------------------
st.header("1) Upload historical data (monthly)")

st.markdown("### Upload Your Workforce Data")
st.info("""
Please upload a CSV file with the following columns:
- `date` (YYYY-MM-DD)
- `output` (numeric, e.g., units produced or sales)
- `employees` (integer)
Optional columns for advanced metrics:
- `labor_cost` (numeric)
- `revenue` (numeric)
Make sure the data is chronological (oldest to latest) and monthly.
""")

def make_sample_data(num_months):
    dates = pd.date_range(end=datetime.now(), periods=num_months, freq='M')
    output = np.linspace(1000, 1000 + num_months * 50, num_months) + np.random.normal(0, 50, num_months)
    employees = (output / 100).astype(int)
    labor_cost = employees * 5000 + np.random.normal(0, 1000, num_months)
    revenue = output * 200 + np.random.normal(0, 5000, num_months)
    return pd.DataFrame({
        'date': dates,
        'output': output,
        'employees': employees,
        'labor_cost': labor_cost,
        'revenue': revenue
    })

sample_data = make_sample_data(48)
csv_buffer = io.StringIO()
sample_data.to_csv(csv_buffer, index=False)
st.download_button(
    "Download Sample CSV",
    csv_buffer.getvalue(),
    "sample_workforce_data.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Choose CSV file", type="csv")
required_cols = ['date', 'output', 'employees']

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Please upload a valid dataset.")
        st.stop()
    else:
        st.success("✅ File uploaded successfully!")

    df['date'] = pd.to_datetime(df['date'])
    st.dataframe(df.sort_values('date').tail(12).reset_index(drop=True))

    # -----------------------
    # Compute productivity
    # -----------------------
    st.header("2) Productivity, Required Manpower, and Forecast")
    df = df.sort_values('date').reset_index(drop=True)

    if is_manual_input:
        prod = manual_prod_value
        st.sidebar.write(f"Manual productivity set to: {prod:.2f} output / employee / month")
    else:
        recent = df.tail(recent_months)
        avg_output_month = recent['output'].mean()
        avg_emp_month = recent['employees'].mean()
        if avg_emp_month <= 0:
            st.warning("Average employees is zero or invalid. Please use manual input.")
            prod = manual_prod_value
        else:
            prod = avg_output_month / avg_emp_month
            st.sidebar.write(f"Observed productivity (avg last {recent_months} months): {prod:.2f} output / employee / month")

    latest_emp = int(df['employees'].iloc[-1])
    current_workforce = st.sidebar.number_input("Available workforce (current headcount)", min_value=1, value=latest_emp)
    st.sidebar.write(f"Latest dataset headcount = {latest_emp}")

    # -----------------------
    # Forecasting functions
    # -----------------------
    def forecast_prophet(df_in, months):
        if Prophet is None:
            raise ImportError("Prophet is not installed. Install with `pip install prophet` or use conda.")
        tmp = df_in[['date','output']].rename(columns={'date':'ds','output':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(tmp)
        future = m.make_future_dataframe(periods=months, freq='M')
        fc = m.predict(future)
        res = fc[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'date','yhat':'yhat_output','yhat_lower':'yhat_lower','yhat_upper':'yhat_upper'})
        return res

    def forecast_arima(df_in, months):
        if ARIMA is None:
            raise ImportError("statsmodels not installed or ARIMA not available.")
        series = df_in.set_index('date')['output'].asfreq('M')
        model = ARIMA(series, order=(1,1,1))
        fitted = model.fit()
        pred = fitted.get_forecast(steps=months)
        idx = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='M')
        pred_mean = pred.predicted_mean
        conf = pred.conf_int()
        res = pd.DataFrame({
            'date': idx,
            'yhat_output': pred_mean.values,
            'yhat_lower': conf.iloc[:,0].values,
            'yhat_upper': conf.iloc[:,1].values
        })
        return res

    def forecast_linear(df_in, months):
        df_tmp = df_in.copy()
        df_tmp['t'] = np.arange(len(df_tmp))
        X = df_tmp[['t']]
        y = df_tmp['output']
        lr = LinearRegression().fit(X, y)
        last_t = df_tmp['t'].iloc[-1]
        future_t = np.arange(last_t+1, last_t+months+1)
        yhat = lr.predict(future_t.reshape(-1,1))
        idx = pd.date_range(start=df_tmp['date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=months, freq='M')
        res = pd.DataFrame({'date': idx, 'yhat_output': yhat, 'yhat_lower': yhat*0.9, 'yhat_upper': yhat*1.1})
        return res

    # Run forecasting
    if model_choice.startswith("Prophet"):
        try:
            fc_df = forecast_prophet(df, horizon_months)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            fc_df = forecast_linear(df, horizon_months)
    elif model_choice.startswith("ARIMA"):
        try:
            fc_df = forecast_arima(df, horizon_months)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            fc_df = forecast_linear(df, horizon_months)
    else:
        fc_df = forecast_linear(df, horizon_months)

    # Compute required manpower for forecast only
    fc_df['required_manpower'] = fc_df['yhat_output'] / prod
    fc_df['gap'] = fc_df['required_manpower'] - current_workforce
    fc_df['available_workforce'] = current_workforce

    # -----------------------
    # Visualization
    # -----------------------
    st.subheader("Forecast: output (historical + forecast)")
    fig = px.line()
    fig.add_scatter(x=df['date'], y=df['output'], mode='lines+markers', name='Historical output')
    fig.add_scatter(x=fc_df['date'], y=fc_df['yhat_output'], mode='lines+markers', name='Forecasted output')
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Output (units)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Required manpower (forecasted) vs available")
    fig2 = px.line(fc_df, x='date', y='required_manpower', labels={'required_manpower':'Required manpower (headcount)'})
    fig2.add_scatter(x=fc_df['date'], y=fc_df['available_workforce'], mode='lines+markers', name='Available workforce')
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Workforce gap (positive = shortage; negative = excess)")
    fig3 = px.bar(fc_df, x='date', y='gap', labels={'gap':'Gap (required - available)'})
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------
    # Financial impact
    # -----------------------
    st.header("3) Financial impact of staffing errors")

    monthly_salary = avg_annual_salary / 12
    monthly_salary_with_benefits = monthly_salary * (1 + benefits_pct)

    if 'revenue' in df.columns and df['employees'].iloc[-1] != 0:
        rev_per_emp = df['revenue'].iloc[-1] / df['employees'].iloc[-1]
    elif 'revenue' in df.columns and df['employees'].mean() > 0:
        rev_per_emp = df['revenue'].mean() / df['employees'].mean()
    else:
        avg_output = df['output'].mean() if not df.empty else 1
        avg_rev = df['revenue'].mean() if 'revenue' in df.columns and not df['revenue'].empty else avg_output * 100
        avg_emp = df['employees'].mean() if not df['employees'].empty else 1
        rev_per_emp = avg_rev / avg_emp

    st.write(f"Estimated revenue per employee (latest month): {rev_per_emp:,.2f}")

    fc_df['overstaff_cost'] = ((fc_df['available_workforce'] - fc_df['required_manpower']).clip(lower=0) * monthly_salary_with_benefits)
    fc_df['understaff_lost_revenue'] = ((fc_df['required_manpower'] - fc_df['available_workforce']).clip(lower=0) * rev_per_emp * lost_sales_factor)
    fc_df['understaff_lost_profit'] = fc_df['understaff_lost_revenue'] * gross_margin

    total_overstaff_cost = fc_df['overstaff_cost'].sum()
    total_understaff_lost_revenue = fc_df['understaff_lost_revenue'].sum()
    total_understaff_lost_profit = fc_df['understaff_lost_profit'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total forecast shortage (headcount, sum of positive gaps)", f"{int(fc_df['gap'].clip(lower=0).sum()):d}")
    col2.metric("Total overstaff cost (currency units)", f"{total_overstaff_cost:,.0f}")
    col3.metric("Total lost revenue due to shortage", f"{total_understaff_lost_revenue:,.0f}")
    col4.metric("Total lost profit (approx)", f"{total_understaff_lost_profit:,.0f}")

    # -----------------------
    # Cost Details Table (historical + forecast)
    # -----------------------
    hist_table = df[['date','employees']].rename(columns={'employees':'available_workforce'})
    hist_table['required_manpower'] = np.nan
    hist_table['gap'] = np.nan
    hist_table['overstaff_cost'] = np.nan
    hist_table['understaff_lost_revenue'] = np.nan
    hist_table['understaff_lost_profit'] = np.nan

    cost_table = pd.concat([hist_table, fc_df[['date','required_manpower','available_workforce','gap','overstaff_cost','understaff_lost_revenue','understaff_lost_profit']]], ignore_index=True)
    st.subheader("Cost details by month")
    st.dataframe(cost_table.round(2))

# -----------------------
# Interpretation & next steps
# -----------------------
if uploaded_file is not None and 'fc_df' in locals():
    st.header("4) Interpretation & HR Insights (Dynamic Analysis)")
    avg_gap = fc_df['gap'].mean()
    avg_required = fc_df['required_manpower'].mean()
    avg_available = fc_df['available_workforce'].mean()

    if avg_gap > 0:
        st.subheader("📉 Workforce Shortage Detected")
        st.markdown(f"""
        - On average, your organization is short by **{avg_gap:.1f} employees per month**.
        - Forecast indicates an upcoming shortage in workforce capacity, which may impact productivity and service delivery.
        - **Recommended Actions:** Initiate recruitment drives, consider contract staffing, or increase employee productivity through training and upskilling.
        """)
    elif avg_gap < 0:
        st.subheader("📈 Potential Overstaffing Observed")
        st.markdown(f"""
        - The model predicts a manpower surplus averaging **{abs(avg_gap):.1f} employees per month**.
        - This may lead to higher labor costs and reduced efficiency.
        - **Recommended Actions:** Optimize manpower allocation, adjust hiring pace, or explore cross-department utilization.
        """)
    else:
        st.subheader("⚖️ Workforce Balanced")
        st.markdown("""
        - Current forecasts suggest manpower demand matches available workforce.
        - Continue monitoring productivity levels and financial performance for early signs of deviation.
        """)

    st.subheader("💰 Financial Risk Assessment")
    if total_overstaff_cost > total_understaff_lost_profit:
        st.info(f"""
        The **overstaffing cost ({total_overstaff_cost:,.0f})** exceeds the potential lost profit due to shortages ({total_understaff_lost_profit:,.0f}).
        This indicates that maintaining extra workforce may not be cost-efficient.  
        **Recommendation:** Optimize hiring plans or shift workloads to improve utilization.
        """)
    elif total_overstaff_cost < total_understaff_lost_profit:
        st.warning(f"""
        The **lost profit from understaffing ({total_understaff_lost_profit:,.0f})** is higher than overstaffing costs ({total_overstaff_cost:,.0f}).  
        **Recommendation:** Plan strategic hiring or temporary staffing to safeguard revenue.
        """)
    else:
        st.success("""
        Overstaffing and understaffing costs are currently balanced.
        The organization’s workforce strategy appears financially optimized — maintain current levels while monitoring market shifts.
        """)

    st.markdown("---")
    st.markdown("""
    ### 🧭 Managerial Interpretation:
    This AI-driven model automatically tailors insights to your dataset, showing whether your organization faces a **workforce surplus or shortage** and quantifying its **financial implications**.
    Managers can use this for:
    - **Hiring & Training Decisions** – Align manpower with forecasted output.
    - **Cost Control** – Minimize unnecessary labor expenses.
    - **Scenario Planning** – Test productivity or salary changes and observe impact on future workforce needs.
    """)

    st.markdown("""
    **Note on assumptions:** Productivity is computed as average monthly output per employee (or can be entered manually). 
    Lost-sales factor and gross margin are user inputs and should be informed by operations data.
    """)

else:
    st.info("👆 Please upload a valid dataset and run the forecast to view interpretation insights.")
