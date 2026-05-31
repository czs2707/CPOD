# COPD-HFpEF Comorbidity Risk Predictor

Streamlit web application for predicting Heart Failure with preserved Ejection Fraction (HFpEF) comorbidity risk in COPD patients using an interpretable XGBoost machine learning model.

## Paper Reference

**Cao J, Kang B, Li S, et al.** Development and external validation of an interpretable machine-learning model for HFpEF comorbidity risk in COPD patients. *Respiratory Research*. 2026. DOI: 10.1186/s12931-026-03711-5

## Model Performance

| Metric | Internal Validation | External Validation |
|--------|-------------------|---------------------|
| AUC | 0.920 | 0.819 |
| Accuracy | 0.837 | 0.794 |
| Sensitivity | 0.835 | 0.786 |
| Specificity | 0.838 | 0.949 |
| F1-Score | 0.831 | 0.708 |

## 9 Key Predictive Features

1. **NT-proBNP** - N-terminal pro-B-type natriuretic peptide
2. **RBC** - Red Blood Cell count
3. **FIB** - Fibrinogen
4. **CHO** - Cholesterol
5. **PaO2** - Arterial partial pressure of oxygen
6. **IC** - Inspiratory Capacity
7. **IC% pred** - Inspiratory Capacity % predicted
8. **A wave** - Late diastolic mitral inflow velocity
9. **CAT score** - COPD Assessment Test score

## Deployment on Streamlit Cloud

### Step 1: Prepare Repository
1. Create a new GitHub repository
2. Upload all files from this project to the repository root

### Step 2: Deploy
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set main file path to `app/app.py`
6. Click "Deploy"

### Alternative: Deploy via Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository
5. Set the main file path: `app/app.py`
6. Click "Deploy"

The app will be automatically built and deployed. You will receive a public URL.

## Local Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
cd app
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Application Features

- **Risk Prediction**: Input patient clinical parameters and get HFpEF risk probability
- **SHAP Interpretation**: Understand which features drive the prediction
- **Model Performance**: View ROC curves, calibration curves, confusion matrix, and DCA
- **Clinical Guidance**: Risk stratification with clinical recommendations

## Disclaimer

This tool is for **research and educational purposes only**. It should not replace clinical judgment. Always consult with qualified healthcare professionals for medical decisions.

## License

This project is licensed under the same terms as the original paper (CC BY-NC-ND 4.0).

## Contact

- Dr. Xiaoyan Xie: 3455434521@qq.com
- Dr. Binghua Zhang: reszbh986@fmmu.edu.cn
- Dr. Wei Guo: 465549513@qq.com

**Institution**: Department of Pulmonary and Critical Care Medicine, Xijing 986th Hospital, Fourth Military Medical University, Xi'an, China
