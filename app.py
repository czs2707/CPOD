"""
COPD-HFpEF Comorbidity Risk Prediction Model
Root entry point for Streamlit Cloud deployment
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ========== Load Model ==========
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    model = joblib.load('app/models/xgboost_best_model.pkl')
    return model

@st.cache_data
def load_test_data():
    """Load test data for model performance visualization"""
    X_test = pd.read_csv('app/data/X_test.csv')
    y_test = pd.read_csv('app/data/y_test.csv').values.ravel()
    return X_test, y_test

# Page configuration
st.set_page_config(
    page_title="COPD-HFpEF Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        font-size: 1.2rem;
        font-weight: bold;
        color: #cc0000;
    }
    .risk-moderate {
        background-color: #fff4cc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        font-size: 1.2rem;
        font-weight: bold;
        color: #cc7700;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #44ff44;
        font-size: 1.2rem;
        font-weight: bold;
        color: #008800;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2a6da8;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
    }
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #1f4e79, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar Navigation ==========
st.sidebar.markdown("<div class='sidebar-info'>", unsafe_allow_html=True)
st.sidebar.title("🫁 Navigation")
st.sidebar.markdown("""
**COPD-HFpEF Risk Prediction Model**

This tool predicts HFpEF comorbidity risk in COPD patients using 9 clinical features.
""")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "🔮 Risk Prediction", "📊 Model Performance", "🔍 SHAP Interpretation", "📖 About"]
)

# ========== Home Page ==========
if page == "🏠 Home":
    st.markdown("<div class='main-header'>COPD-HFpEF Comorbidity Risk Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interpretable Machine Learning Model for Early HFpEF Detection in COPD Patients</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>1,550</div>
            <div class='metric-label'>Training Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>9</div>
            <div class='metric-label'>Key Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.920</div>
            <div class='metric-label'>Internal AUC</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.819</div>
            <div class='metric-label'>External AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("📋 Model Overview")
        st.write("""
        This application implements an **XGBoost-based machine learning model** to predict the risk of 
        **Heart Failure with preserved Ejection Fraction (HFpEF)** in patients with **Chronic Obstructive 
        Pulmonary Disease (COPD)**.
        
        **Key Features:**
        - 🎯 **9 clinical predictors** consistently selected by LASSO, Logistic Regression, and Boruta methods
        - 🔍 **SHAP-based interpretability** for transparent clinical decision-making
        - ✅ **Externally validated** on independent cohort (n=69)
        - 📊 **AUC > 0.90** on internal validation
        
        **Clinical Significance:**
        COPD and HFpEF frequently coexist, leading to increased hospitalization and mortality. 
        Early identification of HFpEF risk enables timely intervention and improved patient outcomes.
        """)
        
        st.subheader("🔬 9 Predictive Biomarkers")
        biomarkers = {
            "NT-proBNP": "N-terminal pro-B-type natriuretic peptide - Most important cardiac biomarker",
            "RBC": "Red Blood Cell count",
            "FIB": "Fibrinogen - Inflammatory marker",
            "CHO": "Cholesterol - Protective factor",
            "PaO₂": "Arterial partial pressure of oxygen",
            "IC": "Inspiratory Capacity (L)",
            "IC% pred": "Inspiratory Capacity % predicted",
            "A wave": "Late diastolic mitral inflow velocity (cm/s)",
            "CAT score": "COPD Assessment Test score (0-40)"
        }
        for marker, desc in biomarkers.items():
            st.markdown(f"**{marker}**: {desc}")
    
    with right_col:
        st.subheader("🚀 Quick Start")
        st.info("""
        **To predict HFpEF risk:**
        
        1. Navigate to **🔮 Risk Prediction** in the sidebar
        2. Enter patient's clinical data
        3. Click **Predict Risk** button
        4. View prediction with SHAP explanation
        """)
        
        st.subheader("📚 Citation")
        st.markdown("""
        *Cao J, Kang B, Li S, et al. Development and external validation 
        of an interpretable machine-learning model for HFpEF comorbidity 
        risk in COPD patients. **Respiratory Research**. 2026.*
        """)
        
        st.subheader("⚠️ Disclaimer")
        st.warning("""
        This tool is for **research and educational purposes only**. 
        It should not replace clinical judgment. Always consult with 
        qualified healthcare professionals for medical decisions.
        """)

# ========== Risk Prediction Page ==========
elif page == "🔮 Risk Prediction":
    st.markdown("<div class='main-header'>🔮 HFpEF Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Enter patient clinical parameters to predict HFpEF comorbidity risk</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🩸 Laboratory")
        nt_probnp = st.number_input("NT-proBNP (pg/mL)", min_value=1.0, max_value=30000.0, value=200.0, step=10.0,
                                   help="N-terminal pro-B-type natriuretic peptide")
        st.markdown("<span class='feature-desc'>Normal: <125 pg/mL | Elevated in HF</span>", unsafe_allow_html=True)
        
        rbc = st.number_input("RBC (×10¹²/L)", min_value=1.0, max_value=8.0, value=4.2, step=0.1,
                             help="Red Blood Cell count")
        st.markdown("<span class='feature-desc'>Normal: 4.0-5.5 ×10¹²/L</span>", unsafe_allow_html=True)
        
        fib = st.number_input("Fibrinogen (g/L)", min_value=0.5, max_value=15.0, value=3.2, step=0.1,
                             help="Inflammatory marker")
        st.markdown("<span class='feature-desc'>Normal: 2.0-4.0 g/L</span>", unsafe_allow_html=True)
        
        cho = st.number_input("Cholesterol (mmol/L)", min_value=0.5, max_value=15.0, value=3.7, step=0.1,
                             help="Total cholesterol")
        st.markdown("<span class='feature-desc'>Normal: <5.2 mmol/L</span>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("🫁 Pulmonary Function")
        pao2 = st.number_input("PaO₂ (mmHg)", min_value=20.0, max_value=150.0, value=70.0, step=1.0,
                              help="Arterial partial pressure of oxygen")
        st.markdown("<span class='feature-desc'>Normal: 80-100 mmHg</span>", unsafe_allow_html=True)
        
        ic = st.number_input("IC (L)", min_value=0.1, max_value=8.0, value=2.0, step=0.1,
                            help="Inspiratory Capacity")
        st.markdown("<span class='feature-desc'>Normal: ~3.5 L (males)</span>", unsafe_allow_html=True)
        
        ic_pct = st.number_input("IC% predicted (%)", min_value=5.0, max_value=150.0, value=65.0, step=1.0,
                                help="Inspiratory Capacity % of predicted")
        st.markdown("<span class='feature-desc'>Normal: >80%</span>", unsafe_allow_html=True)
    
    with col3:
        st.subheader("🫀 Cardiac / Symptoms")
        a_wave = st.number_input("A wave (cm/s)", min_value=20.0, max_value=200.0, value=85.0, step=1.0,
                                help="Late diastolic mitral inflow velocity")
        st.markdown("<span class='feature-desc'>Normal: <80 cm/s</span>", unsafe_allow_html=True)
        
        cat_score = st.number_input("CAT Score", min_value=0, max_value=40, value=22, step=1,
                                   help="COPD Assessment Test (0-40)")
        st.markdown("<span class='feature-desc'>0=Best, 40=Worst | ≥20 high impact</span>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    predict_col, _, _ = st.columns([1, 1, 1])
    with predict_col:
        predict_btn = st.button("🚀 Predict HFpEF Risk", use_container_width=True)
    
    if predict_btn:
        try:
            model = load_model()
            
            input_data = pd.DataFrame({
                'NT_proBNP': [nt_probnp], 'RBC': [rbc], 'FIB': [fib], 'CHO': [cho],
                'PaO2': [pao2], 'IC': [ic], 'IC_pct_pred': [ic_pct], 'A_wave': [a_wave], 'CAT_score': [cat_score]
            })
            
            probability = model.predict_proba(input_data)[0, 1]
            prediction = model.predict(input_data)[0]
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                st.subheader("📊 Prediction Result")
                
                if probability >= 0.7:
                    risk_class = "HIGH RISK"
                    risk_color = "risk-high"
                    risk_advice = "⚠️ **Immediate cardiology referral recommended.** High probability of HFpEF comorbidity."
                elif probability >= 0.4:
                    risk_class = "MODERATE RISK"
                    risk_color = "risk-moderate"
                    risk_advice = "🔍 **Further cardiac evaluation advised.** Consider echocardiography and cardiology consultation."
                else:
                    risk_class = "LOW RISK"
                    risk_color = "risk-low"
                    risk_advice = "✅ **Continue routine COPD management.** Regular monitoring of cardiac function."
                
                st.markdown(f"""
                <div class='{risk_color}'>
                    HFpEF Risk Probability: {probability:.1%}<br>
                    Risk Class: {risk_class}
                </div>
                """, unsafe_allow_html=True)
                
                st.write(risk_advice)
                st.progress(float(probability))
                st.caption(f"Probability: {probability:.4f} (threshold: 0.5)")
            
            with result_col2:
                st.subheader("🔍 Key Contributing Factors")
                
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_vals[0], base_values=explainer.expected_value,
                    data=input_data.iloc[0], feature_names=input_data.columns
                ), show=False)
                plt.title('SHAP Explanation for This Patient', fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            st.subheader("📝 Clinical Interpretation")
            interp_col1, interp_col2 = st.columns(2)
            with interp_col1:
                st.markdown("""
                **🫁 Pulmonary Factors:**
                - **CAT Score ≥20**: Significantly associated with increased cardiovascular events
                - **Reduced IC**: Signals lung hyperinflation, an early COPD indicator
                - **Low PaO₂**: Chronic hypoxia promotes pulmonary vascular remodeling
                """)
            with interp_col2:
                st.markdown("""
                **🫀 Cardiac Factors:**
                - **Elevated NT-proBNP**: Strongest predictor; reflects cardiac wall stress
                - **Elevated A wave**: Indicates increased LV end-diastolic pressure
                - **Altered CHO**: May reflect immune dysfunction and cachexia
                """)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# ========== Model Performance Page ==========
elif page == "📊 Model Performance":
    st.markdown("<div class='main-header'>📊 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Comprehensive evaluation of the XGBoost prediction model</div>", unsafe_allow_html=True)
    
    try:
        X_test, y_test = load_test_data()
        model = load_model()
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
        
        st.subheader("📈 Performance Metrics (Internal Validation)")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        
        metrics_data = [("AUC", f"{auc:.3f}"), ("Accuracy", f"{acc:.3f}"), ("Sensitivity", f"{sens:.3f}"),
                       ("Specificity", f"{spec:.3f}"), ("PPV", f"{ppv:.3f}"), ("NPV", f"{npv:.3f}"), ("F1-Score", f"{f1:.3f}")]
        
        for col, (label, value) in zip([m1, m2, m3, m4, m5, m6, m7], metrics_data):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("📉 ROC Curve")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color='#1f4e79', linewidth=3, label=f'XGBoost (AUC = {auc:.3f})')
        ax1.fill_between(fpr, tpr, alpha=0.15, color='#1f4e79')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Reference Line')
        ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
        ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.01, 1.01])
        ax1.set_ylim([-0.01, 1.01])
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col_cm, col_cal = st.columns(2)
        
        with col_cm:
            st.subheader("🔢 Confusion Matrix")
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       xticklabels=['COPD Only', 'COPD+HFpEF'],
                       yticklabels=['COPD Only', 'COPD+HFpEF'],
                       annot_kws={'size': 16, 'weight': 'bold'})
            ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
            ax2.set_ylabel('True Label', fontsize=11)
            ax2.set_xlabel('Predicted Label', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        
        with col_cal:
            st.subheader("📐 Calibration Curve")
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
            ax3.plot(mean_pred, frac_pos, 's-', color='#1f4e79', linewidth=2.5, markersize=10, label='XGBoost')
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
            ax3.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
            ax3.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax3.set_ylabel('Fraction of Positives', fontsize=12)
            ax3.set_title('Calibration Curve', fontsize=13, fontweight='bold')
            ax3.legend(loc='lower right', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1])
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("📊 Decision Curve Analysis (DCA)")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        thresholds = np.arange(0, 1.01, 0.01)
        n = len(y_test)
        prevalence = np.mean(y_test)
        
        def calc_net_benefit(y_true, y_prob, thresh):
            w = thresh / (1 - thresh)
            pred_pos = (y_prob >= thresh).astype(int)
            tp = np.sum((pred_pos == 1) & (y_true == 1))
            fp = np.sum((pred_pos == 1) & (y_true == 0))
            return (tp / n) - (fp / n) * w
        
        nb_model = [calc_net_benefit(y_test, y_prob, t) for t in thresholds]
        nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
        nb_none = [0] * len(thresholds)
        
        ax4.plot(thresholds, nb_model, color='#1f4e79', linewidth=2.5, label='XGBoost Model')
        ax4.plot(thresholds, nb_all, 'k--', linewidth=1.5, alpha=0.6, label='Treat All')
        ax4.plot(thresholds, nb_none, 'k-', linewidth=1.5, alpha=0.6, label='Treat None')
        ax4.fill_between(thresholds, nb_model, nb_none, where=[m > 0 for m in nb_model], 
                         alpha=0.1, color='green', label='Net Benefit Zone')
        ax4.set_xlabel('Threshold Probability', fontsize=12)
        ax4.set_ylabel('Net Benefit', fontsize=12)
        ax4.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 1])
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        
        st.info("""
        **DCA Interpretation:** The model provides clinical net benefit in the threshold probability 
        range of approximately 0.45 to 0.85, indicating good clinical utility for HFpEF risk screening.
        """)
    
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")

# ========== SHAP Interpretation Page ==========
elif page == "🔍 SHAP Interpretation":
    st.markdown("<div class='main-header'>🔍 SHAP Model Interpretation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Understanding feature contributions using SHAP (SHapley Additive exPlanations)</div>", unsafe_allow_html=True)
    
    try:
        X_test, y_test = load_test_data()
        model = load_model()
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        st.subheader("📊 Global Feature Importance (Mean |SHAP|)")
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, color='#1f4e79')
        plt.title('Mean Absolute SHAP Values', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        st.info("""
        **NT-proBNP** is the most influential predictor, followed by **A wave** and **CAT score**. 
        Elevated NT-proBNP levels are strongly associated with HFpEF comorbidity in COPD patients.
        """)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("🐝 SHAP Distribution (Beeswarm Plot)")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP Value Distribution by Feature', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        st.info("""
        **Color interpretation:** Red = high feature value, Blue = low feature value. 
        Features are sorted by importance. Points on the right push prediction toward HFpEF (+).
        """)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("👤 Individual Patient SHAP Analysis")
        patient_options = [f"Patient {i} ({'HFpEF' if y_test[i] == 1 else 'No HFpEF'})" for i in range(len(y_test))]
        selected_patient = st.selectbox("Select a patient:", range(len(patient_options)), 
                                       format_func=lambda i: patient_options[i])
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[selected_patient], base_values=explainer.expected_value,
            data=X_test.iloc[selected_patient], feature_names=X_test.columns
        ), show=False)
        plt.title(f'SHAP Waterfall Plot - Patient #{selected_patient}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        
        prob = model.predict_proba(X_test.iloc[[selected_patient]])[0, 1]
        st.info(f"Predicted HFpEF probability: **{prob:.4f} ({prob:.1%})**")
        
    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")

# ========== About Page ==========
elif page == "📖 About":
    st.markdown("<div class='main-header'>📖 About This Application</div>", unsafe_allow_html=True)
    
    st.subheader("Study Background")
    st.write("""
    **Chronic Obstructive Pulmonary Disease (COPD)** and **Heart Failure with preserved Ejection Fraction (HFpEF)** 
    frequently coexist, leading to increased hospitalization, mortality, and healthcare burden. Early identification 
    of HFpEF risk in COPD patients is critical for timely intervention.
    
    This application implements the machine learning model described in the paper published in **Respiratory Research** (2026):
    
    > *Cao J, Kang B, Li S, et al. Development and external validation of an interpretable machine-learning model 
    > for HFpEF comorbidity risk in COPD patients.*
    """)
    
    st.subheader("Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Study Design:**
        - Retrospective study of 1,550 COPD patients
        - 803 COPD-only, 747 COPD+HFpEF
        - 70/30 train-test split with fixed seed=2025
        - External validation cohort (n=69)
        
        **Feature Selection:**
        - LASSO regression with 10-fold CV
        - Logistic regression with RFECV
        - Boruta random forest (500 iterations)
        - Only features confirmed by ALL 3 methods were selected
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning Models:**
        1. Logistic Regression
        2. Support Vector Machine (SVM)
        3. Gradient Boosting Machine (GBM)
        4. Neural Network
        5. Random Forest
        6. **XGBoost** (selected as final model)
        7. k-Nearest Neighbors
        8. AdaBoost
        9. LightGBM
        10. CatBoost
        
        **XGBoost Performance:**
        - Internal AUC: 0.898 (95% CI: 0.867-0.929)
        - External AUC: 0.819 (95% CI: 0.713-0.924)
        """)
    
    st.subheader("9 Key Predictive Features")
    
    features_df = pd.DataFrame({
        'Feature': ['NT-proBNP', 'RBC', 'FIB', 'CHO', 'PaO₂', 'IC', 'IC% pred', 'A wave', 'CAT score'],
        'Description': [
            'N-terminal pro-B-type natriuretic peptide (pg/mL)',
            'Red Blood Cell count (×10¹²/L)',
            'Fibrinogen (g/L)',
            'Cholesterol (mmol/L)',
            'Arterial partial pressure of oxygen (mmHg)',
            'Inspiratory Capacity (L)',
            'Inspiratory Capacity % predicted',
            'Late diastolic mitral inflow velocity (cm/s)',
            'COPD Assessment Test score (0-40)'
        ],
        'Category': ['Cardiac', 'Hematology', 'Inflammation', 'Metabolic', 'ABG', 'PFT', 'PFT', 'Echo', 'Symptom']
    })
    
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    st.subheader("Pathophysiological Insights")
    st.write("""
    **NT-proBNP** emerged as the most influential predictor. Elevated levels reflect:
    - Right ventricular load from hypoxia and pulmonary hypertension
    - Leftward septal displacement
    - Systemic inflammation and chronic hypoxia
    - Left ventricular stiffness and myocardial fibrosis
    
    **Hypoxia** (low PaO₂) represents an independent risk factor:
    - Promotes pulmonary vascular remodeling via HIF-1α pathway
    - Induces systemic inflammation (IL-6, TNF-α)
    - Causes coronary microvascular endothelial dysfunction
    
    **Reduced IC** signals lung hyperinflation, an early COPD indicator that may precede FEV₁ decline.
    
    **High CAT score (≥20)** is associated with increased cardiovascular events through mechanisms 
    involving systemic inflammation and vascular endothelial dysfunction.
    """)
    
    st.subheader("Technical Information")
    st.code("""
Model: XGBoost (eXtreme Gradient Boosting)
Hyperparameters (tuned via 10-fold CV):
  - n_estimators: 200
  - max_depth: 3
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
  - random_state: 2025

Evaluation Metrics:
  - ROC-AUC, Accuracy, Sensitivity, Specificity
  - PPV, NPV, F1-Score
  - Calibration curve (Hosmer-Lemeshow test)
  - Decision Curve Analysis (DCA)
  - SHAP values for interpretability

Software: Python 3.12, scikit-learn, XGBoost, SHAP, Streamlit
    """)
    
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    **This tool is for research and educational purposes only.** It is not intended to replace 
    clinical judgment or professional medical advice. Always consult with qualified healthcare 
    professionals for diagnosis and treatment decisions.
    
    The model was developed and validated on specific patient populations and may not generalize 
    to all clinical settings. Use at your own risk.
    """)
    
    st.subheader("📧 Contact")
    st.write("""
    For questions about this application or the original research, please contact the corresponding authors:
    - **Dr. Xiaoyan Xie**: 3455434521@qq.com
    - **Dr. Binghua Zhang**: reszbh986@fmmu.edu.cn
    - **Dr. Wei Guo**: 465549513@qq.com
    
    **Institution:** Department of Pulmonary and Critical Care Medicine, 
    Xijing 986th Hospital, Fourth Military Medical University, Xi'an, China
    """)
