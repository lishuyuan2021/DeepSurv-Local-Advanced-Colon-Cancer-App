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
# 1. 定义网络结构 (4层, 32节点)
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
# 2. 资源加载函数
# ==============================================================================
@st.cache_resource
def load_resources():
    model = DeepSurvNet()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    # 加载 1/3/5/10年基准生存率
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
st.title("🩺 DeepSurv Local Advanced Colon Cancer Individualized Prognosis Tool")
st.markdown("Developed via DeepSurv (Deep Learning) based on SEER database.")

with st.sidebar:
    st.header("Patient Characteristics")
    age = st.slider("Age (years)", 18, 100, 65)
    nodes_pos = st.number_input("Regional Nodes Positive", 0, 50, 0)
    nodes_exam = st.number_input("Regional Nodes Examined", 1, 100, 15)
    
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
# 4. 预测与绘图逻辑 (请确保这部分全部在按钮缩进内)
# ==============================================================================
if st.sidebar.button("🚀 Run Prognostic Analysis", type="primary"):
    # --- A. 构建输入向量 (省略中... 确保这部分代码有缩进) ---
    input_vec = np.zeros(27)
    # ... (你的那些 if therapy == 'AC' 等逻辑)

    # --- B. 执行预测 ---
    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)
    with torch.no_grad():
        log_hazard = model(input_tensor).item()
        # 🟢 确保在这里定义了 relative_risk
        relative_risk = np.exp(log_hazard) 

    # --- C. 计算生存概率 (必须紧跟在 relative_risk 定义之后，且保持缩进) ---
    try:
        # 尝试作为字符串读取 Key (12, 36, 60, 120)
        surv_1y = (base_surv['12'] ** relative_risk) * 100
        surv_3y = (base_surv['36'] ** relative_risk) * 100
        surv_5y = (base_surv['60'] ** relative_risk) * 100
        surv_10y = (base_surv['120'] ** relative_risk) * 100
    except KeyError:
        # 如果 CSV 读取后 Key 变成了整数，则用这个补丁
        surv_1y = (base_surv[12] ** relative_risk) * 100
        surv_3y = (base_surv[36] ** relative_risk) * 100
        surv_5y = (base_surv[60] ** relative_risk) * 100
        surv_10y = (base_surv[120] ** relative_risk) * 100

    # --- D. 展示结果 (同样需要缩进) ---
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
        # SHAP 绘图逻辑...
        explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
        shap_values = explainer.shap_values(input_tensor)
        
        fig, ax = plt.subplots()
        # 强制压缩维度防止报错
        sv = np.squeeze(shap_values)
        exp = shap.Explanation(values=sv, base_values=explainer.expected_value[0], 
                               data=input_vec, feature_names=feature_list)
        shap.plots.waterfall(exp, max_display=10, show=False)
        st.pyplot(fig)

st.markdown("---")
st.caption("Note: This tool is for research purpose only.")