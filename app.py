# ============================================
# Manufacturing Machine Failure Prediction
# Smart Lookup + Manual Fallback
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------- Page Config ----------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ============================================
# LOAD MODEL ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

# ============================================
# LOAD MACHINE DATABASE (with cleaning)
# ============================================
@st.cache_data
def load_machine_db():
    df = pd.read_csv('machine_data.csv')
    # Clean column names and values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    # Drop any duplicate rows on Machine ID + Type
    df = df.drop_duplicates(subset=['Machine ID', 'Type'], keep='first')
    return df

model, scaler, feature_names = load_artifacts()
machine_db = load_machine_db()

# ---------- Header ----------
st.title("⚙️ Manufacturing Machine Failure Prediction")
st.markdown("""
Enter the **Machine ID** and **Machine Type**. If the machine exists in our
database, its readings load automatically. Otherwise, you can enter them manually.
""")
st.divider()

# ============================================
# SIDEBAR — MACHINE IDENTIFICATION
# ============================================
st.sidebar.header("🏭 Machine Identification")

machine_id = st.sidebar.text_input(
    "Machine ID",
    value="",
    placeholder="e.g., M-001, PUMP-42",
    help="Enter the Machine ID"
).strip()

machine_type = st.sidebar.selectbox(
    "Machine Type",
    options=["L", "M", "H"],
    help="L = Low quality, M = Medium quality, H = High quality"
)

st.sidebar.divider()

# ============================================
# LOOKUP LOGIC
# ============================================
matched = pd.DataFrame()
if machine_id:
    matched = machine_db[
        (machine_db['Machine ID'] == machine_id) &
        (machine_db['Type'] == machine_type)
    ]

is_known = not matched.empty

# ============================================
# SIDEBAR — OPERATING PARAMETERS
# ============================================
st.sidebar.header("🔧 Operating Parameters")

if is_known:
    # ---- Known machine: auto-load readings ----
    row = matched.iloc[0]
    air_temp = float(row['Air temperature [K]'])
    process_temp = float(row['Process temperature [K]'])
    rot_speed = int(float(row['Rotational speed [rpm]']))
    torque = float(row['Torque [Nm]'])
    tool_wear = int(float(row['Tool wear [min]']))

    st.sidebar.success("📡 Machine found — readings loaded automatically")
    st.sidebar.write(f"**Air Temperature:** {air_temp} K")
    st.sidebar.write(f"**Process Temperature:** {process_temp} K")
    st.sidebar.write(f"**Rotational Speed:** {rot_speed} rpm")
    st.sidebar.write(f"**Torque:** {torque} Nm")
    st.sidebar.write(f"**Tool Wear:** {tool_wear} min")

elif machine_id == "":
    # ---- No ID entered yet ----
    st.sidebar.info("⬆️ Enter a Machine ID above to continue")
    air_temp = process_temp = rot_speed = torque = tool_wear = None

else:
    # ---- New machine: manual sliders ----
    st.sidebar.warning("🆕 New machine — enter parameters manually")
    air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
    rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1500, 10)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 260, 100, 1)

# ============================================
# FEATURE ENGINEERING
# ============================================
if machine_id != "" and air_temp is not None:
    temp_diff = process_temp - air_temp
    power = torque * rot_speed
    torque_x_wear = torque * tool_wear
    speed_torque_ratio = rot_speed / (torque + 1)
    wear_rate = tool_wear / (rot_speed + 1)

    type_L = 1 if machine_type == "L" else 0
    type_M = 1 if machine_type == "M" else 0
    type_H = 1 if machine_type == "H" else 0

    input_dict = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Temp_Difference': temp_diff,
        'Power': power,
        'Torque_x_Wear': torque_x_wear,
        'Speed_Torque_Ratio': speed_torque_ratio,
        'Wear_Rate': wear_rate,
        'Type_H': type_H,
        'Type_L': type_L,
        'Type_M': type_M,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ============================================
# MAIN LAYOUT
# ============================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Machine Parameters")

    if machine_id == "":
        st.info("⬆️ Enter a Machine ID in the sidebar to get started.")
    else:
        st.markdown(f"**Machine ID:** `{machine_id}`")
        st.markdown(f"**Machine Type:** `{machine_type}`")
        st.markdown("---")

        readings_display = pd.DataFrame({
            "Parameter": [
                "Air Temperature",
                "Process Temperature",
                "Rotational Speed",
                "Torque",
                "Tool Wear",
                "Temp Difference",
                "Power",
                "Torque × Wear"
            ],
            "Value": [
                f"{air_temp:.2f} K",
                f"{process_temp:.2f} K",
                f"{rot_speed} rpm",
                f"{torque:.2f} Nm",
                f"{tool_wear} min",
                f"{temp_diff:.2f} K",
                f"{power:.0f} W",
                f"{torque_x_wear:.0f}"
            ]
        })
        st.dataframe(readings_display, hide_index=True, use_container_width=True)

        if is_known:
            st.success("📡 Readings loaded automatically from database")
        else:
            st.info("✏️ New machine — readings entered manually")

with col2:
    st.subheader("🔮 Failure Prediction")

    if machine_id == "":
        st.info("Prediction will be available once a Machine ID is entered.")
    else:
        if st.button("Predict Failure Risk", type="primary", use_container_width=True):
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown(f"**Machine:** `{machine_id}` | **Time:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
            st.markdown("---")

            # ---- Risk thresholds ----
            if probability >= 0.50:
                risk_emoji, risk_label = "🔴", "CRITICAL"
                st.error(f"⚠️ **Machine {machine_id} — CRITICAL FAILURE RISK**")
            elif probability >= 0.20:
                risk_emoji, risk_label = "🟠", "HIGH"
                st.warning(f"⚠️ **Machine {machine_id} — HIGH FAILURE RISK**")
            elif probability >= 0.05:
                risk_emoji, risk_label = "🟡", "MODERATE"
                st.warning(f"⚠️ **Machine {machine_id} — MODERATE FAILURE RISK**")
            else:
                risk_emoji, risk_label = "🟢", "SAFE"
                st.success(f"✅ **Machine {machine_id} — OPERATING NORMALLY**")

            st.metric("Failure Probability", f"{probability*100:.2f}%")
            st.markdown(f"**Risk Level:** {risk_emoji} **{risk_label}**")

            # ---- Recommended actions ----
            if prediction == 1 or probability >= 0.20:
                st.markdown(f"""
                **Recommended Actions for `{machine_id}`:**
                - 🛑 Schedule immediate maintenance
                - 🔍 Inspect tool wear (currently **{tool_wear} min**)
                - ⚡ Check torque load (currently **{torque} Nm**)
                - 📊 Monitor closely for next 24 hours
                """)
            else:
                st.markdown(f"""
                **Status for `{machine_id}`:**
                - ✅ All parameters within safe range
                - 🔄 Continue regular monitoring
                - 📅 Next scheduled maintenance as planned
                """)

            st.progress(float(probability))

            # ---------- Download Report ----------
            st.divider()
            report_data = {
                "Machine ID": [machine_id],
                "Type": [machine_type],
                "Source": ["Database" if is_known else "Manual Entry"],
                "Timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                "Air Temperature [K]": [air_temp],
                "Process Temperature [K]": [process_temp],
                "Rotational Speed [rpm]": [rot_speed],
                "Torque [Nm]": [torque],
                "Tool Wear [min]": [tool_wear],
                "Prediction": ["FAILURE" if prediction == 1 else "NORMAL"],
                "Risk Level": [risk_label],
                "Failure Probability": [f"{probability*100:.2f}%"]
            }
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Inspection Report (CSV)",
                data=csv,
                file_name=f"inspection_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption(
    "Built with Streamlit • Model: Tuned XGBoost • Dataset: AI4I 2020 Predictive Maintenance"
)