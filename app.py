import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# 自动定位路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. 定义网络结构
# ==============================================================================
class DeepSurvNet(nn.Module):
    def __init__(self):
        super(DeepSurvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 1, bias=False)
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. 资源加载
# ==============================================================================
@st.cache_resource
def load_resources():
    model = DeepSurvNet()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    feature_list = [
        "age", "systemic.sur.seqAC", "systemic.sur.seqNAC+AC", "CEAPositive",
        "One.primary.onlyYes", "First.malignant.pri.Yes", "tumor.depositsPositive",
        "regional.nodes.positive", "Perineural.InvasionYes", "regional.nodes.examined",
        "sexMale", "raceWhite", "raceBlack", "raceAmerican Indian/Alaska Native",
        "primary.siteAscending Colon", "primary.siteHepatic Flexure",
        "primary.siteTransverse Colon", "primary.siteSplenic Flexure",
        "primary.siteDescending Colon", "primary.siteSigmoid Colon",
        "primary.siteRectosigmoid Junction", "T.stageT4a", "T.stageT4b",
        "N.stageN1", "N.stageN2", "GradeModerately differentiated",
        "GradePoorly differentiated/Undifferentiated"
    ]
    return model, scalers, bg_data, base_surv, feature_list

model, scalers, bg_data, base_surv, feature_list = load_resources()

# ==============================================================================
# 3. 页面布局
# ==============================================================================
st.set_page_config(page_title="DeepSurv-LACC Predictor", layout="wide")
st.title("DeepSurv Local Advanced Colon Cancer Individualized Prognosis Tool")
st.markdown("Developed via DeepSurv (Deep Learning) based on SEER database.")

with st.sidebar:
    st.header("Patient Characteristics")
    age = st.slider("Age (years)", 18, 79, 60)
    nodes_pos = st.number_input("Regional Nodes Positive", 0, 45, 0)
    nodes_exam = st.number_input("Regional Nodes Examined", 1, 90, 15)
    
    sex = st.selectbox("Sex", ["Female", "Male"])
    race = st.selectbox("Race", ["Asian or Pacific Islander", "White", "Black", "American Indian/Alaska Native"])
    cea = st.selectbox("CEA Status", ["Negative", "Positive"])
    perineural = st.selectbox("Perineural Invasion", ["No", "Yes"])
    deposits = st.selectbox("Tumor Deposits", ["Negative", "Positive"])
    
    st.subheader("Pathological Info")
    t_stage = st.selectbox("T Stage", ["T3", "T4a", "T4b"])
    n_stage = st.selectbox("N Stage", ["N0", "N1", "N2"])
    grade = st.selectbox("Grade", ["Well differentiated", "Moderately differentiated", "Poorly differentiated/Undifferentiated"])
    
    st.subheader("Other Factors")
    site = st.selectbox("Primary Site", ["Cecum", "Ascending Colon", "Hepatic Flexure", "Transverse Colon", 
                                        "Splenic Flexure", "Descending Colon", "Sigmoid Colon", "Rectosigmoid Junction"])
    therapy = st.selectbox("Systemic Therapy Sequence", ["NAC", "AC", "NAC+AC"])
    primary_only = st.selectbox("One Primary Only", ["No", "Yes"])
    first_malig = st.selectbox("First Malignant Primary", ["No", "Yes"])

# ==============================================================================
# 4. 预测与绘图逻辑
# ==============================================================================
if st.sidebar.button("🚀 Run Prognostic Analysis", type="primary"):
    # --- A. 构建输入向量 (27维) ---
    input_vec = np.zeros(27)
    
    # 填充连续变量并标准化
    input_vec[0] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    input_vec[7] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    input_vec[9] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    
    # 填充所有哑变量（确保此处逻辑完整！）
    if therapy == "AC": input_vec[1] = 1
    elif therapy == "NAC+AC": input_vec[2] = 1
    
    if cea == "Positive": input_vec[3] = 1
    if primary_only == "Yes": input_vec[4] = 1
    if first_malig == "Yes": input_vec[5] = 1
    if deposits == "Positive": input_vec[6] = 1
    if perineural == "Yes": input_vec[8] = 1
    if sex == "Male": input_vec[10] = 1
    
    if race == "White": input_vec[11] = 1
    elif race == "Black": input_vec[12] = 1
    elif race == "American Indian/Alaska Native": input_vec[13] = 1
    
    site_map = {"Ascending Colon": 14, "Hepatic Flexure": 15, "Transverse Colon": 16, 
                "Splenic Flexure": 17, "Descending Colon": 18, "Sigmoid Colon": 19, "Rectosigmoid Junction": 20}
    if site in site_map: input_vec[site_map[site]] = 1
    
    if t_stage == "T4a": input_vec[21] = 1
    elif t_stage == "T4b": input_vec[22] = 1
    
    if n_stage == "N1": input_vec[23] = 1
    elif n_stage == "N2": input_vec[24] = 1
    
    if grade == "Moderately differentiated": input_vec[25] = 1
    elif grade == "Poorly differentiated/Undifferentiated": input_vec[26] = 1

    # --- B. 执行预测 ---
    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)
    with torch.no_grad():
        log_hazard = model(input_tensor).item()
        relative_risk = np.exp(log_hazard)

    # --- C. 计算生存概率 ---
    try:
        surv_1y = (base_surv['12'] ** relative_risk) * 100
        surv_3y = (base_surv['36'] ** relative_risk) * 100
        surv_5y = (base_surv['60'] ** relative_risk) * 100
        surv_10y = (base_surv['120'] ** relative_risk) * 100
    except KeyError:
        surv_1y = (base_surv[12] ** relative_risk) * 100
        surv_3y = (base_surv[36] ** relative_risk) * 100
        surv_5y = (base_surv[60] ** relative_risk) * 100
        surv_10y = (base_surv[120] ** relative_risk) * 100

    # --- D. 展示结果 ---
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.success("#### Survival Probability")
        st.metric("1-Year Survival", f"{surv_1y:.1f}%")
        st.metric("3-Year Survival", f"{surv_3y:.1f}%")
        st.metric("5-Year Survival", f"{surv_5y:.1f}%")
        st.metric("10-Year Survival", f"{surv_10y:.1f}%")
        st.write(f"**Individual Risk Score:** {log_hazard:.4f}")

    with col2:
        st.info("#### Individualized Explanation")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_values = explainer.shap_values(input_tensor)
            
            fig, ax = plt.subplots()
            sv = np.squeeze(shap_values)
            exp = shap.Explanation(values=sv, base_values=explainer.expected_value[0], 
                                   data=input_vec, feature_names=feature_list)
            shap.plots.waterfall(exp, max_display=10, show=False)
            st.pyplot(fig)

st.markdown("---")
st.caption("Note: This tool is for research purpose only.")