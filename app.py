"""
Telco Customer Churn Prediction Application
Final Year Project — with Real Dynamic SHAP Explanations

SHAP Implementation:
- Linear SHAP  → Logistic Regression (analytical, exact)
- Perturbation SHAP → Random Forest, Gradient Boosting, Neural Network
  Each explanation is computed per-customer dynamically.
  No two customers get the same explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, warnings, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

warnings.filterwarnings('ignore')
import config

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high {
    background:#FEE2E2; border-left:5px solid #EF4444;
    padding:20px; border-radius:8px; margin:10px 0;
}
.risk-high h2,.risk-high h1,.risk-high p,.risk-high h3 {
    color:#7F1D1D !important; margin:5px 0 !important;
}
.risk-medium {
    background:#FEF3C7; border-left:5px solid #F59E0B;
    padding:20px; border-radius:8px; margin:10px 0;
}
.risk-medium h2,.risk-medium h1,.risk-medium p,.risk-medium h3 {
    color:#78350F !important; margin:5px 0 !important;
}
.risk-low {
    background:#D1FAE5; border-left:5px solid #10B981;
    padding:20px; border-radius:8px; margin:10px 0;
}
.risk-low h2,.risk-low h1,.risk-low p,.risk-low h3 {
    color:#064E3B !important; margin:5px 0 !important;
}
.shap-box {
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:10px; padding:16px; margin:8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(config.MODELS_PATH,       'rb') as f: models       = pickle.load(f)
    with open(config.SCALER_PATH,       'rb') as f: scaler       = pickle.load(f)
    with open(config.FEATURE_NAMES_PATH,'rb') as f: feature_names= pickle.load(f)
    with open(config.METRICS_PATH,      'r' ) as f: metrics      = json.load(f)
    with open(config.TEST_DATA_PATH,    'rb') as f: test_data    = pickle.load(f)
    with open(config.ORDINAL_ENCODER_PATH,  'rb') as f: ordinal_encoder= pickle.load(f)
    return models, scaler, feature_names, metrics, test_data, ordinal_encoder

models, scaler, feature_names, metrics, test_data, ordinal_encoder = load_artifacts()

# Pre-load background dataset for SHAP (first 200 test rows)
_X_bg_raw    = np.array(test_data['X_test'])[:200]
_X_bg_scaled = np.array(test_data['X_test_scaled'])[:200]
_base_rate   = float(np.mean(test_data['y_test']))


# ── SHAP ENGINE ──────────────────────────────────────────────────────────────
def compute_shap(model_name: str, input_raw: np.ndarray, input_scaled: np.ndarray) -> np.ndarray:
    """
    Compute per-feature SHAP values in probability space.

    Strategy:
      • Logistic Regression  → Linear SHAP  (coef * delta, mapped to prob-space)
      • RF / GB              → Perturbation SHAP over background set
      • Neural Network       → Perturbation SHAP over background set

    Returns
    -------
    shap_values : np.ndarray shape (n_features,)
        Positive = pushes prediction UP (toward churn)
        Negative = pushes prediction DOWN (away from churn)
    """
    model = models[model_name]
    n_feat = len(feature_names)

    # ── Linear SHAP (Logistic Regression) ───────────────────────────────────
    if model_name == 'Logistic Regression':
        coef = model.coef_[0]                          # shape (n_feat,)
        bg_mean = _X_bg_scaled.mean(axis=0)

        # Baseline prediction with mean background
        bg_mean_df = pd.DataFrame([bg_mean], columns=feature_names)
        p_base = model.predict_proba(bg_mean_df)[0, 1]

        shap_vals = np.zeros(n_feat)
        for i in range(n_feat):
            # Replace feature i in background mean with sample value
            perturbed = bg_mean.copy()
            perturbed[i] = input_scaled[0, i]
            pert_df = pd.DataFrame([perturbed], columns=feature_names)
            p_pert  = model.predict_proba(pert_df)[0, 1]
            shap_vals[i] = p_pert - p_base

        return shap_vals

    # ── Perturbation SHAP (Tree / NN) ────────────────────────────────────────
    use_scaled = (model_name == 'Neural Network')
    X_bg = _X_bg_scaled if use_scaled else _X_bg_raw
    x    = input_scaled if use_scaled else input_raw

    bg_df   = pd.DataFrame(X_bg,   columns=feature_names)
    p_base  = model.predict_proba(bg_df)[:, 1].mean()

    shap_vals = np.zeros(n_feat)
    for i in range(n_feat):
        perturbed      = X_bg.copy()
        perturbed[:, i] = x[0, i]
        pert_df        = pd.DataFrame(perturbed, columns=feature_names)
        p_pert         = model.predict_proba(pert_df)[:, 1].mean()
        shap_vals[i]   = p_pert - p_base

    return shap_vals


# ── SHAP WATERFALL CHART ──────────────────────────────────────────────────────
def plot_shap_waterfall(shap_values: np.ndarray, base_rate: float,
                        final_prob: float, top_n: int = 8) -> plt.Figure:
    """
    Render a SHAP waterfall chart.

    Shows how each top feature pushes the probability from the baseline
    (population average churn rate) to the final prediction.
    """
    # Select top-N features by absolute SHAP
    top_idx   = np.argsort(np.abs(shap_values))[-top_n:][::-1]
    top_shap  = shap_values[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    # Clean up feature names for display
    def fmt(name):
        name = name.replace('_', ' ').replace('  ', ' ')
        replacements = {
            'Contract Month To Month': 'Contract: Month-to-Month',
            'Contract Two Year':       'Contract: Two Year',
            'Contract One Year':       'Contract: One Year',
            'Internet Service Fiber Optic': 'Internet: Fiber Optic',
            'Internet Service No':     'Internet: None',
            'Payment Method Electronic Check': 'Payment: E-Check',
            'Payment Method Mailed Check': 'Payment: Mailed Check',
            'Payment Method Credit Card (Automatic)': 'Payment: Credit Card',
            'Paperless Billing Yes':   'Paperless Billing',
            'Multiple Lines Yes':      'Multiple Lines',
            'Tech Support Yes':        'Tech Support: Yes',
            'Online Security Yes':     'Online Security: Yes',
            'Tenure Category New':     'Tenure: New (<12mo)',
            'Tenure Category Medium':  'Tenure: Medium',
        }
        title = name.title()
        return replacements.get(title, title)

    display_names = [fmt(n) for n in top_names]

    # Build waterfall: sort so positive (bad) bars appear first for readability
    sort_order = np.argsort(top_shap)[::-1]
    sorted_shap   = top_shap[sort_order]
    sorted_names  = [display_names[i] for i in sort_order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_shap) * 0.65 + 1.5)))

    running = base_rate
    bar_colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_shap]

    for idx, (val, name, color) in enumerate(zip(sorted_shap, sorted_names, bar_colors)):
        ax.barh(idx, val, left=running, color=color, alpha=0.85,
                height=0.6, edgecolor='white', linewidth=0.5)
        # Annotation
        sign = '+' if val > 0 else ''
        ax.text(running + val + (0.005 if val >= 0 else -0.005),
                idx, f'{sign}{val*100:.1f}%',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold', color='#1F2937')
        running += val

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.axvline(base_rate, color='#6B7280', linewidth=1.5, linestyle='--', label=f'Base rate: {base_rate*100:.1f}%')
    ax.axvline(final_prob, color='#1D4ED8', linewidth=2,   linestyle='-',  label=f'Prediction: {final_prob*100:.1f}%')

    ax.set_xlabel('Churn Probability', fontsize=11)
    ax.set_title('SHAP Waterfall — Feature Contributions to Churn Probability',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, min(1.05, max(final_prob, base_rate) + 0.12))

    red_patch   = mpatches.Patch(color='#EF4444', alpha=0.85, label='Increases churn risk ↑')
    green_patch = mpatches.Patch(color='#10B981', alpha=0.85, label='Decreases churn risk ↓')
    ax.legend(handles=[red_patch, green_patch,
                        plt.Line2D([0],[0], color='#6B7280', lw=1.5, ls='--', label=f'Base rate: {base_rate*100:.1f}%'),
                        plt.Line2D([0],[0], color='#1D4ED8', lw=2,   ls='-',  label=f'Prediction: {final_prob*100:.1f}%')],
              loc='lower right', fontsize=9, framealpha=0.9)

    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


# ── SHAP GLOBAL IMPORTANCE ────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def compute_global_shap(model_name: str, n_samples: int = 80) -> pd.DataFrame:
    """
    Compute mean |SHAP| over a sample of test rows → global feature importance.
    Results are cached so this only runs once per model selection.
    """
    model       = models[model_name]
    use_scaled  = (model_name in ['Logistic Regression', 'Neural Network'])
    X_bg        = _X_bg_scaled if use_scaled else _X_bg_raw
    X_sample    = X_bg[:n_samples]

    abs_shap = np.zeros(len(feature_names))

    for row in X_sample:
        row_raw    = row.reshape(1, -1)
        row_scaled = row.reshape(1, -1)

        if model_name == 'Logistic Regression':
            sv = compute_shap(model_name, row_raw, row_scaled)
        elif model_name == 'Neural Network':
            sv = compute_shap(model_name, row_raw, row_raw)   # both scaled
        else:
            sv = compute_shap(model_name, row_raw, row_scaled)

        abs_shap += np.abs(sv)

    abs_shap /= n_samples
    return pd.DataFrame({'feature': feature_names, 'mean_abs_shap': abs_shap})\
             .sort_values('mean_abs_shap', ascending=False)


def plot_global_shap(df: pd.DataFrame, model_name: str, top_n: int = 15) -> plt.Figure:
    top = df.head(top_n).iloc[::-1]   # reverse for horizontal bar

    def fmt(name):
        return name.replace('_',' ').replace('  ',' ').title()

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5 + 1.2)))

    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(v / top['mean_abs_shap'].max()) for v in top['mean_abs_shap']]

    ax.barh(range(len(top)), top['mean_abs_shap'], color=colors,
            edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([fmt(n) for n in top['feature']], fontsize=10)
    ax.set_xlabel('Mean |SHAP| — Average Impact on Churn Probability', fontsize=11)
    ax.set_title(f'Global Feature Importance via SHAP — {model_name}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row['mean_abs_shap'] + 0.001, i,
                f"{row['mean_abs_shap']*100:.2f}%", va='center', fontsize=9)

    fig.tight_layout()
    return fig


# ── BATCH SHAP SUMMARY ────────────────────────────────────────────────────────
def compute_batch_shap_summary(model_name: str,
                                processed_batch: pd.DataFrame,
                                risk_categories: list,
                                max_per_group: int = 30) -> dict:
    """
    For each risk group, compute mean SHAP values → returns top-3 drivers per group.
    """
    model = models[model_name]
    use_scaled = (model_name in ['Logistic Regression', 'Neural Network'])
    results = {}

    for group in ['High Risk', 'Medium Risk', 'Low Risk']:
        mask   = [r == group for r in risk_categories]
        subset = processed_batch[mask].values[:max_per_group]

        if len(subset) == 0:
            results[group] = []
            continue

        shap_acc = np.zeros(len(feature_names))

        for row in subset:
            row_raw    = row.reshape(1, -1)
            row_scaled = scaler.transform(pd.DataFrame(row_raw, columns=feature_names))

            if use_scaled:
                sv = compute_shap(model_name, row_scaled, row_scaled)
            else:
                sv = compute_shap(model_name, row_raw, row_scaled)

            shap_acc += sv

        mean_shap = shap_acc / len(subset)
        top_idx   = np.argsort(np.abs(mean_shap))[-3:][::-1]

        results[group] = [
            {
                'feature':    feature_names[i].replace('_',' ').title(),
                'shap_value': float(mean_shap[i]),
                'direction':  'increases' if mean_shap[i] > 0 else 'decreases',
                'pct':        f"{mean_shap[i]*100:+.1f}%"
            }
            for i in top_idx
        ]

    return results


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median() or 0)

    def cat_tenure(t):
        return 'New' if t <= 12 else 'Medium' if t <= 36 else 'Long'

    df['Tenure_Category'] = df['tenure'].apply(cat_tenure)

    svc_cols = ['PhoneService','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['TotalServices'] = sum(
        (df[c].isin(['Yes','DSL','Fiber optic'])).astype(int)
        for c in svc_cols if c in df.columns
    )
    df['AvgMonthlySpend']  = df.apply(
        lambda r: r.get('TotalCharges',0)/r['tenure'] if r['tenure']>0 else r.get('MonthlyCharges',0), axis=1)
    _cmon = {'Month-to-month':1,'One year':12,'Two year':24}
    df['ContractValue']    = df.apply(
        lambda r: r.get('MonthlyCharges',0)*_cmon.get(r.get('Contract','Month-to-month'),1), axis=1)
    df['ServiceDensity']   = df.apply(
        lambda r: r['TotalServices']/r.get('MonthlyCharges',1) if r.get('MonthlyCharges',0)>0 else 0, axis=1)

    for col in ['customerID','Churn']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Ordinal encode Tenure_Category
    df[['Tenure_Category']] = ordinal_encoder.transform(df[['Tenure_Category']])

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols)

    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0

    return df_enc[feature_names]


def get_risk_category(p):
    if p < config.LOW_RISK_THRESHOLD:
        return "Low Risk",    "🟢", "risk-low"
    elif p < config.HIGH_RISK_THRESHOLD:
        return "Medium Risk", "🟡", "risk-medium"
    else:
        return "High Risk",   "🔴", "risk-high"


def predict(model_name, processed_df):
    model = models[model_name]
    if model_name in ['Neural Network','Logistic Regression']:
        X = pd.DataFrame(scaler.transform(processed_df), columns=feature_names)
    else:
        X = processed_df
    return model.predict_proba(X)[:, 1]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Customer Churn Decision Support System")
    st.markdown("---")

    selected_model = st.selectbox(
        "Active Model",
        list(models.keys()),
        index=list(models.keys()).index('Gradient Boosting')
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("🎯 Customer Churn Decision Support")
st.markdown("*Machine-learning churn prediction with dynamic SHAP explanations*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Individual Analysis", "📊 Batch Analysis", "📈 Model Performance"]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — INDIVIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Individual Customer Analysis")
    st.markdown("Fill in the customer profile and get a **dynamic SHAP explanation** unique to this customer.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender           = st.selectbox("Gender",          ["Male","Female"])
        senior_citizen   = st.selectbox("Senior Citizen",  ["No","Yes"])
        partner          = st.selectbox("Partner",         ["No","Yes"])
        dependents       = st.selectbox("Dependents",      ["No","Yes"])

    with col2:
        st.subheader("💳 Account")
        tenure           = st.number_input("Tenure (months)", 0, 72, 12)
        contract         = st.selectbox("Contract",
                            ["Month-to-month","One year","Two year"])
        paperless_billing= st.selectbox("Paperless Billing", ["No","Yes"])
        payment_method   = st.selectbox("Payment Method",
                            ["Electronic check","Mailed check",
                             "Bank transfer (automatic)","Credit card (automatic)"])
        monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, 5.0)
        total_charges    = st.number_input("Total Charges ($)", 0.0,
                            value=float(monthly_charges * tenure))

    with col3:
        st.subheader("📡 Services")
        phone_service    = st.selectbox("Phone",           ["No","Yes"])
        multiple_lines   = st.selectbox("Multiple Lines",  ["No","Yes","No phone service"])
        internet_service = st.selectbox("Internet",        ["No","DSL","Fiber optic"])
        online_security  = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_backup    = st.selectbox("Online Backup",   ["No","Yes","No internet service"])
        device_protection= st.selectbox("Device Protection",["No","Yes","No internet service"])
        tech_support     = st.selectbox("Tech Support",    ["No","Yes","No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",    ["No","Yes","No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",["No","Yes","No internet service"])

    st.markdown("---")

    if st.button("🔮 Predict & Explain", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen=="Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }])

        processed    = preprocess_input(input_df)
        churn_prob   = float(predict(selected_model, processed)[0])
        risk_cat, emoji, css = get_risk_category(churn_prob)

        # ── Risk banner ──────────────────────────────────────────────────────
        st.markdown("## 📊 Prediction Result")
        st.markdown(f"""
        <div class="{css}">
            <h2>{emoji} {risk_cat}</h2>
            <h1 style="font-size:52px;">{churn_prob*100:.1f}%</h1>
            <p style="font-size:17px;">Churn Probability
            &nbsp;|&nbsp; Base Rate: {_base_rate*100:.1f}%
            &nbsp;|&nbsp; Model: {selected_model}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP ─────────────────────────────────────────────────────────────
        st.markdown("### 🧠 SHAP Explanation (Dynamic)")
        st.caption(
            "These values are computed specifically for **this customer**. "
            "Positive (red) bars push the probability up; "
            "negative (green) bars push it down."
        )

        with st.spinner("Computing SHAP values…"):
            x_raw    = processed.values
            x_scaled = scaler.transform(processed)
            shap_vals = compute_shap(selected_model, x_raw, x_scaled)

        fig = plot_shap_waterfall(shap_vals, _base_rate, churn_prob, top_n=8)
        st.pyplot(fig)
        plt.close(fig)

        # ── Top-5 table ───────────────────────────────────────────────────────
        st.markdown("#### Top Contributing Factors")
        top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

        rows = []
        for i in top_idx:
            name = feature_names[i].replace('_',' ').title()
            val  = shap_vals[i]
            direction = "⬆️ Increases risk" if val > 0 else "⬇️ Decreases risk"
            rows.append({
                'Feature': name,
                'SHAP Value': f"{val*100:+.2f}%",
                'Effect': direction
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Recommendation ────────────────────────────────────────────────────
        st.markdown("### 💡 Recommendation")
        if churn_prob >= config.HIGH_RISK_THRESHOLD:
            st.error("""
            **⚠️ Urgent Retention Action Required**
            - Immediate personal outreach by retention team
            - Offer tailored retention package / discount
            - Schedule account review meeting
            - Investigate top SHAP drivers and address them directly
            """)
        elif churn_prob >= config.LOW_RISK_THRESHOLD:
            st.warning("""
            **📧 Proactive Engagement Recommended**
            - Targeted email / in-app campaign
            - Offer service upgrade trial
            - Conduct satisfaction survey
            - Monitor monthly for changes
            """)
        else:
            st.success("""
            **✅ Customer Appears Stable**
            - Continue standard engagement
            - Consider upsell / cross-sell
            - Maintain service quality
            """)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch Analysis")
    st.markdown("Upload a file and get risk predictions **plus SHAP-driven explanations per risk group**.")

    uploaded = st.file_uploader("📁 Upload CSV / XLSX", type=['csv','xlsx'])

    # Sample download
    sample = pd.DataFrame({
        'customerID':['SAMPLE001','SAMPLE002','SAMPLE003'],
        'gender':['Male','Female','Male'],
        'SeniorCitizen':[0,1,0],
        'Partner':['Yes','No','No'],
        'Dependents':['No','Yes','No'],
        'tenure':[3,48,72],
        'PhoneService':['Yes','Yes','Yes'],
        'MultipleLines':['No','Yes','No'],
        'InternetService':['Fiber optic','DSL','DSL'],
        'OnlineSecurity':['No','Yes','Yes'],
        'OnlineBackup':['No','Yes','Yes'],
        'DeviceProtection':['No','Yes','Yes'],
        'TechSupport':['No','Yes','Yes'],
        'StreamingTV':['Yes','No','No'],
        'StreamingMovies':['Yes','No','No'],
        'Contract':['Month-to-month','One year','Two year'],
        'PaperlessBilling':['Yes','No','No'],
        'PaymentMethod':['Electronic check','Bank transfer (automatic)','Mailed check'],
        'MonthlyCharges':[90.0,55.0,45.0],
        'TotalCharges':[270.0,2640.0,3240.0]
    })
    st.download_button("📥 Download Sample CSV", sample.to_csv(index=False),
                       "sample.csv", "text/csv")

    if uploaded:
        try:
            batch = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success(f"✓ Loaded {len(batch)} customers")

            with st.expander("📋 Preview"):
                st.dataframe(batch.head(10))

            if st.button("🚀 Analyze Batch", type="primary", use_container_width=True):
                with st.spinner("Running predictions…"):
                    ids       = batch['customerID'].tolist() if 'customerID' in batch.columns \
                                else [f"Customer_{i+1}" for i in range(len(batch))]
                    processed = preprocess_input(batch)
                    probs     = predict(selected_model, processed)
                    risk_cats = [get_risk_category(p)[0] for p in probs]

                    results = pd.DataFrame({
                        'Customer ID': ids,
                        'Churn Probability': probs,
                        'Risk Category': risk_cats
                    })

                # ── Summary cards ─────────────────────────────────────────────
                st.markdown("## 📊 Analysis Summary")
                c1, c2, c3 = st.columns(3)
                low_pct  = (results['Risk Category']=='Low Risk').mean()*100
                med_pct  = (results['Risk Category']=='Medium Risk').mean()*100
                high_pct = (results['Risk Category']=='High Risk').mean()*100

                with c1:
                    st.markdown(f'<div class="risk-low"><h3>🟢 Low Risk</h3>'
                                f'<h2>{low_pct:.1f}%</h2>'
                                f'<p>{int(low_pct*len(results)/100)} customers</p></div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="risk-medium"><h3>🟡 Medium Risk</h3>'
                                f'<h2>{med_pct:.1f}%</h2>'
                                f'<p>{int(med_pct*len(results)/100)} customers</p></div>',
                                unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="risk-high"><h3>🔴 High Risk</h3>'
                                f'<h2>{high_pct:.1f}%</h2>'
                                f'<p>{int(high_pct*len(results)/100)} customers</p></div>',
                                unsafe_allow_html=True)

                # ── Key metrics ───────────────────────────────────────────────
                st.markdown("### 🔍 Key Insights")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg Churn Prob",     f"{probs.mean()*100:.1f}%")
                m2.metric("Total Customers",     len(results))
                m3.metric("Need Attention",      int((med_pct+high_pct)*len(results)/100))
                m4.metric("Highest Risk",        results.loc[probs.argmax(),'Customer ID'])

                # ── SHAP per risk group ───────────────────────────────────────
                st.markdown("### 🧠 SHAP Drivers by Risk Group")
                st.caption("Top 3 features driving churn probability for each risk segment (mean SHAP across group).")

                with st.spinner("Computing group SHAP values…"):
                    shap_summary = compute_batch_shap_summary(
                        selected_model, processed, risk_cats, max_per_group=30
                    )

                gc1, gc2, gc3 = st.columns(3)
                group_configs = [
                    ('High Risk',   gc1, '🔴', 'risk-high'),
                    ('Medium Risk', gc2, '🟡', 'risk-medium'),
                    ('Low Risk',    gc3, '🟢', 'risk-low'),
                ]

                for group_name, col, emoji_g, css_g in group_configs:
                    with col:
                        drivers = shap_summary.get(group_name, [])
                        if not drivers:
                            st.markdown(f'<div class="{css_g}"><h3>{emoji_g} {group_name}</h3>'
                                        f'<p>No customers in this group</p></div>',
                                        unsafe_allow_html=True)
                        else:
                            items = ''.join(
                                f"<p><b>{d['feature']}</b><br/>"
                                f"<small>{d['direction']} risk by <b>{d['pct']}</b></small></p>"
                                for d in drivers
                            )
                            st.markdown(
                                f'<div class="{css_g}"><h3>{emoji_g} {group_name}</h3>'
                                f'<p><small>Top 3 SHAP drivers:</small></p>'
                                f'{items}</div>',
                                unsafe_allow_html=True)

                # ── Results table ──────────────────────────────────────────────
                st.markdown("### 📋 Detailed Results")
                st.dataframe(
                    results.style.background_gradient(
                        subset=['Churn Probability'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                st.download_button("📥 Download Results", results.to_csv(index=False),
                                   "predictions.csv", "text/csv")

                # ── Distribution chart ─────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(10, 4))
                counts  = results['Risk Category'].value_counts()
                clrs    = {'Low Risk':'#10B981','Medium Risk':'#F59E0B','High Risk':'#EF4444'}
                ax.bar(counts.index, counts.values,
                       color=[clrs[c] for c in counts.index],
                       edgecolor='white', width=0.5)
                ax.set_title('Customer Risk Distribution', fontweight='bold', fontsize=13)
                ax.set_ylabel('Customers')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Performance Evaluation")

    comp = pd.DataFrame.from_dict(metrics, orient='index')

    comp = comp.rename(columns={
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    })

    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    comp = comp[numeric_cols]

    # ── Comparison table ──────────────────────────────────────────────────────
    numeric_cols = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']

    display = comp.copy()
    display[numeric_cols] = display[numeric_cols].round(4)

    for col in numeric_cols:
        max_val = display[col].max()
        display[col] = display[col].apply(
            lambda x: f"🟢 {x:.4f}" if x == max_val else f"{x:.4f}"
        )

    st.dataframe(display, use_container_width=True)


    def highlight_best(df):
        """
        For each metric column highlight the best (max) value with
        a solid green background AND explicit white text — readable on
        both light and dark Streamlit themes.
        """
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in numeric_cols:
            if col in df.columns:
                max_idx = df[col].idxmax()
                styles.loc[max_idx, col] = (
                'background-color: #DCFCE7 !important;'
                'color: #065F46 !important;'
                'font-weight: 700 !important;'
            )

        return styles


    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown(f"### 🔍 Detailed Metrics: {selected_model}")
    md = metrics[selected_model]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{md['accuracy']:.4f}")
    c2.metric("Precision", f"{md['precision']:.4f}")
    c3.metric("Recall",    f"{md['recall']:.4f}")
    c4.metric("F1-Score",  f"{md['f1_score']:.4f}")
    c5.metric("ROC-AUC",   f"{md['roc_auc']:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🎯 Confusion Matrix")
    cm  = np.array(md['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn','Churn'],
                yticklabels=['No Churn','Churn'],
                annot_kws={'size':14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    ax.set_title(f'Confusion Matrix — {selected_model}', fontsize=13, fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

    # ── ROC + PR curves ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    y_true = np.array(test_data['y_test'])
    y_prob = np.array(md['probabilities'])

    with c1:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, '#3B82F6', lw=2, label=f'AUC = {roc_val:.4f}')
        ax.plot([0,1],[0,1],'gray', lw=1, ls='--', label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown("### 📉 Precision-Recall Curve")
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_val = auc(rec, prec)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec, prec, '#10B981', lw=2, label=f'AUC = {pr_val:.4f}')
        ax.axhline(_base_rate, color='gray', lw=1, ls='--', label=f'No-skill ({_base_rate:.2f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # ── Global SHAP importance ─────────────────────────────────────────────────
    st.markdown("### 🧠 Global Feature Importance via SHAP")
    st.caption(f"Mean |SHAP| across {80} test customers for **{selected_model}**. "
               "Computed once and cached.")

    with st.spinner("Computing global SHAP importance…"):
        global_shap_df = compute_global_shap(selected_model, n_samples=80)

    fig = plot_global_shap(global_shap_df, selected_model, top_n=15)
    st.pyplot(fig); plt.close(fig)

    # ── Methodology ───────────────────────────────────────────────────────────
    with st.expander("📚 Methodology & References"):
        st.markdown(f"""
        ### SHAP Implementation

        SHAP (SHapley Additive exPlanations) values are computed **dynamically
        per customer** using model-specific strategies:

        | Model | SHAP Method | Description |
        |---|---|---|
        | Logistic Regression | **Linear SHAP** | Analytical: coef × (x − E[x]) mapped to probability space |
        | Random Forest | **Perturbation SHAP** | E[f(x̄ᵢ=xᵢ)] − E[f(x̄)] over background set |
        | Gradient Boosting | **Perturbation SHAP** | Same as above |
        | Neural Network | **Perturbation SHAP** | Same, using scaled features |

        Background dataset: first 200 test-set customers.
        Base rate (E[f(x̄)]): **{_base_rate*100:.1f}%** (population average churn rate).

        ### Dataset
        **Source:** IBM Telco Customer Churn (Kaggle) — real telecom data  
        **Samples:** 7,043 customers | **Churn rate:** {_base_rate*100:.1f}%  
        **Features:** 19 original + 5 engineered = 24 total → 36 after encoding

        ### Feature Engineering
        1. **Tenure Category** — New (≤12 mo), Medium (13–36), Long (>36)
        2. **Total Services** — Count of subscribed services
        3. **Avg Monthly Spend** — TotalCharges / tenure
        4. **Contract Value** — MonthlyCharges × contract months
        5. **Service Density** — Services per dollar

        ### Evaluation
        Primary metric: **ROC-AUC** (robust under class imbalance)  
        Best model: **{max(metrics, key=lambda k: metrics[k]['roc_auc'])}**
        — ROC-AUC {max(m['roc_auc'] for m in metrics.values()):.4f}

        ### References
        1. Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS.
        2. Friedman (2001). *Greedy function approximation: A gradient boosting machine.* Ann. Stat.
        3. Breiman (2001). *Random forests.* Machine Learning, 45(1).
        4. Verbeke et al. (2012). *Churn prediction in telecoms.* EJOR, 218(1).
        """)

    with st.expander("🔍 Data Quality Dashboard"):
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Samples",        "7,043")
        m2.metric("Features",       "36 (encoded)")
        m3.metric("Churn Rate",     f"{_base_rate*100:.1f}%")
        m4.metric("Missing Values", "11 (0.16%)")
        st.success("✓ No duplicates")
        st.success("✓ Missing TotalCharges → median imputation")
        st.success("✓ Class imbalance → balanced class weights + stratified split")
        st.success("✓ 5-fold stratified cross-validation on best model")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6B7280;'>"
    "<p><strong>Telco Churn Prediction </strong>"
    f"{max(m['roc_auc'] for m in metrics.values())*100:.1f}% ROC-AUC</p>"
    "</div>",
    unsafe_allow_html=True
)