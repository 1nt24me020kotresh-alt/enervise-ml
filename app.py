# EnerVise — CNC Energy Optimization Web App
# Kotresh, Aurko, Parv | ASM Student Chapter Bengaluru
# IEEE Research Project 2025

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="EnerVise — CNC Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODEL FILES ──
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, '..', 'models')
    with open(os.path.join(
            model_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(
            model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(
            model_dir, 'feature_columns.json'), 'r') as f:
        features = json.load(f)
    with open(os.path.join(
            model_dir, 'results_summary.json'), 'r') as f:
        summary = json.load(f)
    return model, scaler, features, summary

model, scaler, feature_columns, summary = load_model()

# ── HEADER ──
st.markdown("""
<div style='background: linear-gradient(
    135deg, #0D2137, #2E75B6);
    padding: 30px; border-radius: 12px;
    margin-bottom: 20px;'>
    <h1 style='color:white; margin:0;
               font-size:2.5rem;'>
        ⚡ EnerVise
    </h1>
    <p style='color:#A8C8E8; margin:0;
              font-size:1.1rem;'>
        AI-Powered CNC Energy Optimization
        for Sustainable Manufacturing
    </p>
    <p style='color:#7AAAC8; margin:0;
              font-size:0.85rem;'>
        ASM Student Chapter Bengaluru |
        IEEE Research Project 2025
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons"
    "/thumb/2/21/IEEE_logo.svg/200px-IEEE_logo.svg.png",
    width=80)
st.sidebar.title("About EnerVise")
st.sidebar.markdown("""
**EnerVise** uses machine learning to predict and
optimize CNC machining energy consumption.

**Model:** Decision Tree Regressor
**R² Score:** 0.9724
**Dataset:** UC Michigan CNC Mill
**Training samples:** 14,016

**Research Team:**
Kotresh · Aurko · Parv
ASM Student Chapter, Bengaluru
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
sc1, sc2 = st.sidebar.columns(2)
sc1.metric("R² Score", "0.9724")
sc2.metric("RMSE", "0.0131")

# ── MAIN TABS ──
tab1, tab2, tab3 = st.tabs([
    "⚡ Optimize Parameters",
    "📊 Model Results",
    "💰 Business Impact"
])

# ════════════════════════════════════════
# TAB 1 — OPTIMIZER
# ════════════════════════════════════════
with tab1:
    st.header("CNC Process Parameter Optimizer")
    st.markdown(
        "Adjust your machine settings below. "
        "EnerVise predicts energy consumption in "
        "real time and recommends optimal parameters.")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── LEFT: Sliders ──
    with col_left:
        st.subheader("🔧 Your Current Settings")

        feedrate = st.slider(
            "Feedrate (mm/min)",
            min_value=3.0, max_value=20.0,
            value=6.7, step=0.1,
            help="Speed at which cutting tool moves")

        clamp_pressure = st.slider(
            "Clamp Pressure (bar)",
            min_value=2.5, max_value=4.0,
            value=3.4, step=0.1,
            help="Pressure holding the workpiece")

        spindle_velocity = st.slider(
            "Spindle Velocity (RPM)",
            min_value=0.0, max_value=53.8,
            value=20.0, step=0.1,
            help="Actual spindle rotational speed")

        s1_current = st.slider(
            "S1 Current Feedback (A)",
            min_value=0.0, max_value=75.4,
            value=15.9, step=0.1,
            help="Spindle motor current — "
                 "primary energy driver (95.98% importance)")

    # ── RIGHT: Predictions ──
    with col_right:
        st.subheader("📈 Live Prediction Results")

        # Build input using all slider values
        input_values = {
            'feedrate':            feedrate,
            'clamp_pressure':      clamp_pressure,
            'S1_ActualVelocity':   spindle_velocity,
            'X1_CurrentFeedback':  -0.46,
            'Y1_CurrentFeedback':  -0.10,
            'Z1_CurrentFeedback':  0.0,
            'S1_CurrentFeedback':  s1_current,
            'M1_CURRENT_FEEDRATE': feedrate
        }

        # Build optimal input
        optimal_values = {
            'feedrate':
                summary['optimal_feedrate'],
            'clamp_pressure':
                summary['optimal_clamp_pressure'],
            'S1_ActualVelocity':   spindle_velocity,
            'X1_CurrentFeedback':  -0.46,
            'Y1_CurrentFeedback':  -0.10,
            'Z1_CurrentFeedback':  0.0,
            'S1_CurrentFeedback':  s1_current,
            'M1_CURRENT_FEEDRATE':
                summary['optimal_feedrate']
        }

        # Convert to arrays in correct feature order
        input_arr = np.array(
            [input_values[c]
             for c in feature_columns]
        ).reshape(1, -1)

        optimal_arr = np.array(
            [optimal_values[c]
             for c in feature_columns]
        ).reshape(1, -1)

        # Scale and predict
        input_scaled   = scaler.transform(input_arr)
        optimal_scaled = scaler.transform(optimal_arr)

        current_power = max(
            model.predict(input_scaled)[0], 0)
        optimal_power = max(
            model.predict(optimal_scaled)[0], 0)

        if current_power > 0:
            reduction = ((current_power - optimal_power)
                         / current_power * 100)
        else:
            reduction = 0.0

        # ── Metrics ──
        m1, m2 = st.columns(2)
        m1.metric(
            "Current Power",
            f"{current_power:.4f} units",
            help="Predicted power at your settings")
        m2.metric(
            "Optimized Power",
            f"{optimal_power:.4f} units",
            delta=f"-{abs(reduction):.1f}%",
            delta_color="inverse")

        # ── Spindle Current Gauge ──
        st.markdown("---")
        st.markdown("**⚡ Spindle Current Load**")
        pct = int((s1_current / 75.4) * 100)
        st.progress(pct)

        if s1_current > 50:
            st.warning(
                f"⚠️ High spindle current ({pct}% of max). "
                "Reduce cutting depth or use sharper tooling "
                "to lower energy consumption.")
        elif s1_current > 25:
            st.info(
                f"ℹ️ Moderate spindle current ({pct}% of max). "
                "Within normal operating range.")
        else:
            st.success(
                f"✅ Low spindle current ({pct}% of max). "
                "Machine operating efficiently.")

        st.caption(
            "S1 Current drives 95.98% of energy consumption "
            "— monitor this closely.")

        # ── Recommendation Box ──
        st.markdown("---")
        st.subheader("✅ Recommended Settings")
        st.success(f"""
**Set these parameters for minimum energy:**

🔹 Feedrate: **{summary['optimal_feedrate']:.2f} mm/min**
🔹 Clamp Pressure: **{summary['optimal_clamp_pressure']:.2f} bar**

Expected energy reduction: **5.21%**
Basis: 200,000-point random search optimization
        """)

        # ── Annual Savings ──
        st.markdown("---")
        st.subheader("💰 Annual Savings (Medium VMC)")
        rated_kw    = 15.0
        saved_kw    = rated_kw * 0.0521
        energy_yr   = saved_kw * 16 * 300
        cost_yr     = energy_yr * 8.5
        co2_yr      = energy_yr * 0.82

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Energy Saved",
                   f"{energy_yr:,.0f} kWh/yr")
        sm2.metric("Cost Saved",
                   f"₹{cost_yr:,.0f}/yr")
        sm3.metric("CO₂ Reduced",
                   f"{co2_yr:.0f} kg/yr")

# ════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════
with tab2:
    st.header("ML Model Comparison Results")
    st.markdown(
        "All four models trained and evaluated on "
        "17,520 active cutting rows from 18 real "
        "CNC milling experiments.")

    # Results table
    results_data = {
        'Model': [
            'Linear Regression',
            'Decision Tree ✅',
            'Random Forest',
            'XGBoost'],
        'R² Score': [0.9490, 0.9724, 0.9719, 0.9705],
        'RMSE':     [0.017862, 0.013146,
                     0.013254, 0.013585],
        'MAE':      [0.011847, 0.009127,
                     0.009165, 0.008944],
        'Train Time (s)': [0.06, 0.07, 1.74, 0.31]
    }
    results_df = pd.DataFrame(results_data)

    def highlight_best(row):
        if 'Decision Tree' in row['Model']:
            return ['background-color: #D4EDDA;'
                    'font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(
        results_df.style.apply(
            highlight_best, axis=1),
        use_container_width=True)

    st.markdown("---")

    # Charts
    c1, c2 = st.columns(2)
    colors = ['#e74c3c', '#27ae60',
              '#2980b9', '#8e44ad']
    model_names = [
        'Linear\nRegression', 'Decision\nTree',
        'Random\nForest', 'XGBoost']

    with c1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bars1 = ax1.bar(
            model_names,
            results_data['R² Score'],
            color=colors, alpha=0.85,
            edgecolor='black', linewidth=0.5)
        ax1.set_ylim([0.93, 0.98])
        ax1.set_title('R² Score — Higher is Better',
                      fontweight='bold')
        ax1.set_ylabel('R² Score')
        for bar, val in zip(
                bars1, results_data['R² Score']):
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0002,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with c2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bars2 = ax2.bar(
            model_names,
            results_data['RMSE'],
            color=colors, alpha=0.85,
            edgecolor='black', linewidth=0.5)
        ax2.set_title('RMSE — Lower is Better',
                      fontweight='bold')
        ax2.set_ylabel('RMSE')
        for bar, val in zip(
                bars2, results_data['RMSE']):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0001,
                f'{val:.5f}',
                ha='center', va='bottom',
                fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Feature importance
    st.markdown("---")
    st.subheader(
        "🔍 Key Finding — Feature Importance")

    feat_imp = {
        'S1_CurrentFeedback': 0.9598,
        'S1_ActualVelocity':  0.0352,
        'feedrate':           0.0017,
        'Y1_CurrentFeedback': 0.0014,
        'X1_CurrentFeedback': 0.0012,
        'clamp_pressure':     0.0005,
        'M1_CURRENT_FEEDRATE':0.0002,
        'Z1_CurrentFeedback': 0.0000,
    }

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    names_fi  = list(feat_imp.keys())
    values_fi = list(feat_imp.values())
    col_fi = ['#27ae60' if i == 0 else '#2980b9'
              for i in range(len(names_fi))]
    ax3.barh(names_fi[::-1], values_fi[::-1],
             color=col_fi[::-1], alpha=0.85,
             edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Importance Score',
                   fontsize=12)
    ax3.set_title(
        'Feature Importance — Which Parameters '
        'Drive CNC Energy Consumption Most?',
        fontweight='bold', fontsize=13)
    for i, (name, val) in enumerate(
            zip(names_fi[::-1], values_fi[::-1])):
        ax3.text(val + 0.005, i,
                 f'{val:.4f}',
                 va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.info("""
**Key Research Finding:**
S1_CurrentFeedback (spindle motor current) accounts
for **95.98%** of feature importance — identifying
spindle current as the primary driver of CNC energy
consumption. This finding suggests that real-time
spindle current monitoring alone can serve as a
reliable energy proxy for SME manufacturers.
    """)

    # Actual vs Predicted note
    st.markdown("---")
    st.subheader("📌 Research Summary")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Training Samples", "14,016")
    col_s2.metric("Test Samples", "3,504")
    col_s3.metric("Total Experiments", "18")

    col_s4, col_s5, col_s6 = st.columns(3)
    col_s4.metric("Best Model", "Decision Tree")
    col_s5.metric("R² Score", "0.9724")
    col_s6.metric("Energy Reduction", "5.21%")

# ════════════════════════════════════════
# TAB 3 — BUSINESS IMPACT
# ════════════════════════════════════════
with tab3:
    st.header("Business Impact Analysis")
    st.markdown(
        "Calculate financial and environmental savings "
        "for your specific manufacturing setup.")

    st.markdown("---")

    bi1, bi2 = st.columns(2)

    with bi1:
        machine_type = st.selectbox(
            "Select your CNC machine type:",
            ["Small CNC Mill (3.5 kW) — Light Duty",
             "Medium VMC (15 kW) — ACE Designer, BFW",
             "Heavy VMC (25 kW) — Mazak, DMG Mori"])

        num_machines = st.slider(
            "Number of machines in your shop:",
            min_value=1, max_value=50,
            value=10, step=1)

        shifts = st.radio(
            "Shifts per day:",
            ["Single shift (8 hrs)",
             "Double shift (16 hrs)"])

    with bi2:
        elec_cost = st.slider(
            "Electricity cost (₹/kWh):",
            min_value=5.0, max_value=12.0,
            value=8.5, step=0.5,
            help="KERC industrial tariff ~₹8.5/kWh")

        working_days = st.slider(
            "Working days per year:",
            min_value=200, max_value=365,
            value=300, step=10)

    # Calculate
    power_map = {
        "Small CNC Mill (3.5 kW) — Light Duty": 3.5,
        "Medium VMC (15 kW) — ACE Designer, BFW": 15.0,
        "Heavy VMC (25 kW) — Mazak, DMG Mori": 25.0
    }
    shift_map = {
        "Single shift (8 hrs)": 8,
        "Double shift (16 hrs)": 16
    }

    rated_kw    = power_map[machine_type]
    hrs_day     = shift_map[shifts]
    saved_kw    = rated_kw * 0.0521
    energy_yr   = saved_kw * hrs_day * working_days
    cost_yr     = energy_yr * elec_cost
    co2_yr      = energy_yr * 0.82

    total_cost  = cost_yr  * num_machines
    total_co2   = co2_yr   * num_machines
    total_energy= energy_yr* num_machines

    st.markdown("---")
    st.subheader("📊 Your Savings Dashboard")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Per Machine/Year",
                f"₹{cost_yr:,.0f}")
    r1c2.metric("Total Shop/Year",
                f"₹{total_cost:,.0f}")
    r1c3.metric("CO₂ Saved/Year",
                f"{total_co2:.0f} kg")
    r1c4.metric("Energy Saved/Year",
                f"{total_energy:,.0f} kWh")

    st.markdown("---")
    st.subheader("📈 EnerVise ROI Calculator")

    subscription = 8000 * num_machines * 12
    net_benefit  = total_cost - subscription
    roi_ratio    = (total_cost /
                    max(subscription, 1))

    roi_col1, roi_col2 = st.columns(2)

    with roi_col1:
        st.markdown(f"""
**EnerVise Annual Subscription:**
₹8,000 × {num_machines} machines × 12 months
= **₹{subscription:,}**

**Annual Energy Cost Savings:**
= **₹{total_cost:,}**

**Net Annual Benefit:**
= **₹{net_benefit:,}**

**Return on Investment: {roi_ratio:.1f}x**
        """)

    with roi_col2:
        if roi_ratio >= 2:
            st.success(f"""
✅ **Strong ROI: {roi_ratio:.1f}x**

For every ₹1 spent on EnerVise,
you save ₹{roi_ratio:.1f} in energy.

Payback period: **{12/roi_ratio:.1f} months**
            """)
        elif roi_ratio >= 1:
            st.info(f"""
ℹ️ **Positive ROI: {roi_ratio:.1f}x**

EnerVise pays for itself.
Consider double-shift operation
for stronger returns.
            """)
        else:
            st.warning(
                "Consider a larger machine type "
                "or double-shift operation "
                "for better ROI.")

    st.markdown("---")
    st.subheader("🌍 Environmental Impact")

    trees = total_co2 / 21
    cars  = total_co2 / 2300

    env1, env2, env3 = st.columns(3)
    env1.metric("🌳 Equivalent Trees Planted",
                f"{trees:.0f} trees/yr")
    env2.metric("🚗 Cars Removed from Road",
                f"{cars:.1f} cars/yr")
    env3.metric("♻️ EU CBAM Compliance",
                "Supported")

    st.info("""
**Why this matters:**
The EU Carbon Border Adjustment Mechanism (CBAM)
requires Indian manufacturers exporting to Europe
to report and reduce embodied carbon from 2026.
EnerVise generates automated sustainability reports
to support CBAM compliance and ISO 50001 audits.
    """)

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888;
            font-size:0.85rem; padding:10px;'>
    <strong>EnerVise</strong> |
    ML-Driven CNC Energy Optimization |
    ASM Student Chapter Bengaluru |
    IEEE Research 2025<br>
    Decision Tree Regressor | R² = 0.9724 |
    Trained on UC Michigan CNC Mill Dataset |
    17,520 Training Samples
</div>
""", unsafe_allow_html=True)
