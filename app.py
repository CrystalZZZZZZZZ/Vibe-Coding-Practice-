import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import umap.umap_ as umap
import base64
from io import BytesIO
import time
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="降维方法交互式教学系统",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .formula {
        background-color: #f5f7fa;
        padding: 1rem;
        border-radius: 5px;
        font-family: "Courier New", monospace;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .code-block {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 5px;
        font-family: "Courier New", monospace;
        overflow-x: auto;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'pca_step' not in st.session_state:
    st.session_state.pca_step = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

# 数据加载函数
@st.cache_data
def load_data(dataset_name):
    """加载数据集"""
    if dataset_name == "鸢尾花数据集 (Iris)":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        return X, y, feature_names, target_names, "iris"
    
    elif dataset_name == "手写数字数据集 (Digits)":
        data = datasets.load_digits()
        X = data.data
        y = data.target
        feature_names = [f"像素{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
        return X, y, feature_names, target_names, "digits"
    
    elif dataset_name == "葡萄酒数据集 (Wine)":
        data = datasets.load_wine()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        return X, y, feature_names, target_names, "wine"
    
    elif dataset_name == "随机生成数据":
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = X[:, 1] * 2 + np.random.randn(n_samples) * 0.5  # 创建相关性
        X[:, 2] = X[:, 0] * 1.5 - X[:, 1] * 0.8 + np.random.randn(n_samples) * 0.3
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 二分类标签
        feature_names = [f"特征{i+1}" for i in range(n_features)]
        target_names = ["类别0", "类别1"]
        return X, y, feature_names, target_names, "random"

# PCA推导步骤
def pca_derivation_step(step):
    """显示PCA推导的步骤"""
    steps = [
        {
            "title": "步骤1: 数据标准化",
            "formula": r"X_{\text{standardized}} = \frac{X - \mu}{\sigma}",
            "explanation": "将每个特征减去其均值并除以标准差，确保所有特征具有相同的尺度。",
            "code": """
# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
            """
        },
        {
            "title": "步骤2: 计算协方差矩阵",
            "formula": r"C = \frac{1}{n-1} X^T X",
            "explanation": "协方差矩阵描述了特征之间的线性关系，对角线元素是方差，非对角线元素是协方差。",
            "code": """
# 计算协方差矩阵
n_samples = X_scaled.shape[0]
cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)
            """
        },
        {
            "title": "步骤3: 特征值分解",
            "formula": r"C = V \Lambda V^T",
            "explanation": "对协方差矩阵进行特征值分解，其中V是特征向量矩阵，Λ是对角特征值矩阵。",
            "code": """
# 特征值分解
import numpy as np
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# 按特征值降序排列
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
            """
        },
        {
            "title": "步骤4: 选择主成分",
            "formula": r"k = \arg\max_{k} \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} \geq 0.95",
            "explanation": "选择前k个最大特征值对应的特征向量，通常保留95%的方差。",
            "code": """
# 计算累积解释方差比
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)
# 选择解释95%方差的成分
k = np.argmax(cumulative_variance >= 0.95) + 1
            """
        },
        {
            "title": "步骤5: 投影到低维空间",
            "formula": r"Z = X \cdot V_k",
            "explanation": "将原始数据投影到前k个特征向量张成的子空间上，得到降维后的数据。",
            "code": """
# 投影到主成分空间
V_k = eigenvectors[:, :k]
Z = X_scaled @ V_k
            """
        }
    ]
    
    if step < len(steps):
        step_data = steps[step]
        with st.expander(f"📘 {step_data['title']}", expanded=True):
            st.markdown(f"**数学公式:**")
            st.latex(step_data['formula'])
            st.markdown(f"**解释:** {step_data['explanation']}")
            st.markdown(f"**Python实现:**")
            st.code(step_data['code'], language='python')

# 首页
def home_page():
    st.markdown('<div class="main-header">📉 降维方法交互式教学系统</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎯 学习目标</h3>
        <ul>
            <li>理解降维的核心思想与数学原理</li>
            <li>掌握PCA、t-SNE、UMAP、LDA等经典算法</li>
            <li>通过交互可视化直观理解算法过程</li>
            <li>学会在实际问题中应用降维技术</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">📚 课程大纲</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["PCA", "t-SNE", "UMAP", "LDA"])
        
        with tab1:
            st.markdown("""
            **主成分分析 (PCA)**
            - 线性降维的经典方法
            - 基于方差最大化原理
            - 适用于数据探索和特征提取
            """)
            
        with tab2:
            st.markdown("""
            **t-分布随机邻域嵌入 (t-SNE)**
            - 非线性降维方法
            - 保持局部数据结构
            - 擅长可视化高维数据
            """)
            
        with tab3:
            st.markdown("""
            **均匀流形逼近与投影 (UMAP)**
            - 基于流形学习的最新方法
            - 计算效率高，可扩展性好
            - 保持全局和局部结构
            """)
            
        with tab4:
            st.markdown("""
            **线性判别分析 (LDA)**
            - 监督降维方法
            - 最大化类间距离，最小化类内距离
            - 适用于分类问题的特征提取
            """)
    
    with col2:
        st.markdown('<div class="section-header">🚀 快速开始</div>', unsafe_allow_html=True)
        
        # 数据集选择
        dataset_option = st.selectbox(
            "选择数据集",
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
            index=0
        )
        
        # 降维方法选择
        method_option = st.selectbox(
            "选择降维方法",
            ["PCA - 主成分分析", "t-SNE - t分布随机邻域嵌入", "UMAP - 均匀流形逼近", "LDA - 线性判别分析"],
            index=0
        )
        
        if st.button("开始探索", type="primary"):
            st.session_state.current_dataset = dataset_option
            st.session_state.current_method = method_option
            if method_option.startswith("PCA"):
                st.session_state.current_page = "PCA"
            elif method_option.startswith("t-SNE"):
                st.session_state.current_page = "t-SNE"
            elif method_option.startswith("UMAP"):
                st.session_state.current_page = "UMAP"
            elif method_option.startswith("LDA"):
                st.session_state.current_page = "LDA"
            st.rerun()

# PCA页面
def pca_page():
    st.markdown('<div class="main-header">📊 主成分分析 (PCA)</div>', unsafe_allow_html=True)
    
    # 侧边栏控制面板
    with st.sidebar:
        st.markdown("## ⚙️ PCA参数设置")
        
        # 数据集选择
        dataset_option = st.selectbox(
            "数据集",
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
            index=0 if 'current_dataset' not in st.session_state else 
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"].index(st.session_state.current_dataset)
        )
        
        # 加载数据
        X, y, feature_names, target_names, data_type = load_data(dataset_option)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA参数
        n_components = st.slider("主成分数量", 1, min(10, X.shape[1]), 2)
        
        # 是否显示推导过程
        show_derivation = st.checkbox("显示完整数学推导", value=True)
        
        # 是否进行特征值分解
        show_eigen = st.checkbox("显示特征值分解", value=True)
        
    # 主内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["📖 理论基础", "🎨 可视化", "💻 代码实现", "📈 案例应用"])
    
    with tab1:
        st.markdown('<div class="sub-header">📖 PCA数学推导</div>', unsafe_allow_html=True)
        
        if show_derivation:
            # PCA推导步骤控制
            st.markdown("### 推导步骤")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀️ 上一步") and st.session_state.pca_step > 0:
                    st.session_state.pca_step -= 1
            with col2:
                st.progress((st.session_state.pca_step + 1) / 5, text=f"步骤 {st.session_state.pca_step + 1}/5")
            with col3:
                if st.button("下一步 ▶️") and st.session_state.pca_step < 4:
                    st.session_state.pca_step += 1
            
            # 显示当前步骤
            pca_derivation_step(st.session_state.pca_step)
        
        # PCA几何解释
        st.markdown('<div class="section-header">📐 几何解释</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **PCA的核心思想:**
            
            1. **寻找最大方差方向**
               - 第一个主成分是数据方差最大的方向
               - 第二个主成分是与第一个正交且方差次大的方向
            
            2. **坐标旋转**
               - 将原始坐标系旋转到主成分方向
               - 新坐标轴互不相关（正交）
            
            3. **维度压缩**
               - 丢弃方差小的方向
               - 保留大部分信息（方差）
            """)
        
        with col2:
            # 简单示意图
            fig = go.Figure()
            
            # 生成示例数据
            np.random.seed(42)
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta) + np.random.normal(0, 0.1, 100)
            y = 0.5*np.sin(theta) + np.random.normal(0, 0.1, 100)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                name='原始数据',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            # 添加主成分方向
            from sklearn.decomposition import PCA
            pca_temp = PCA(n_components=2)
            X_temp = np.column_stack([x, y])
            pca_temp.fit(X_temp)
            
            # 主成分向量
            for i, (length, vector) in enumerate(zip(pca_temp.explained_variance_, pca_temp.components_)):
                fig.add_trace(go.Scatter(
                    x=[0, vector[0]*length],
                    y=[0, vector[1]*length],
                    mode='lines',
                    name=f'PC{i+1}',
                    line=dict(width=3, color='red' if i==0 else 'orange')
                ))
            
            fig.update_layout(
                title="PCA几何示意图",
                xaxis_title="特征1",
                yaxis_title="特征2",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">🎨 PCA可视化演示</div>', unsafe_allow_html=True)
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 方差解释率
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(explained_variance))],
                y=explained_variance,
                name='各成分解释方差',
                marker_color='#3498db'
            ))
            fig1.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
                y=cumulative_variance,
                name='累积解释方差',
                mode='lines+markers',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8)
            ))
            
            fig1.update_layout(
                title="方差解释率",
                xaxis_title="主成分",
                yaxis_title="解释方差比例",
                height=400
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # 特征值显示
            if show_eigen and hasattr(pca, 'explained_variance_'):
                eigenvalues = pca.explained_variance_
                st.markdown(f"**特征值 (λ):** {', '.join([f'{val:.3f}' for val in eigenvalues])}")
        
        with col2:
            # 降维结果可视化
            if n_components >= 2:
                if len(np.unique(y)) > 1:
                    color_scale = px.colors.qualitative.Set1
                    colors = [color_scale[int(i) % len(color_scale)] for i in y]
                else:
                    colors = '#3498db'
                
                if n_components == 2:
                    fig2 = go.Figure(data=go.Scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(
                            color=colors if isinstance(colors, list) else colors,
                            size=8,
                            opacity=0.7
                        ),
                        text=[f"样本 {i}" for i in range(len(X_pca))],
                        hoverinfo='text'
                    ))
                    
                    fig2.update_layout(
                        title="PCA降维结果 (2D)",
                        xaxis_title="第一主成分",
                        yaxis_title="第二主成分",
                        height=400
                    )
                    
                else:  # 3D
                    fig2 = go.Figure(data=go.Scatter3d(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        z=X_pca[:, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=colors if isinstance(colors, list) else colors,
                            opacity=0.7
                        )
                    ))
                    
                    fig2.update_layout(
                        title="PCA降维结果 (3D)",
                        scene=dict(
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            zaxis_title="PC3"
                        ),
                        height=500
                    )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # 显示解释方差
                total_variance = np.sum(pca.explained_variance_ratio_)
                st.markdown(f"**累积解释方差:** {total_variance:.2%}")
    
    with tab3:
        st.markdown('<div class="sub-header">💻 PCA代码实现</div>', unsafe_allow_html=True)
        
        code_tab1, code_tab2 = st.tabs(["基础实现", "完整示例"])
        
        with code_tab1:
            st.markdown("##### 1. 使用scikit-learn实现")
            st.code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 创建PCA模型
pca = PCA(n_components=2)  # 降到2维

# 3. 拟合和转换数据
X_pca = pca.fit_transform(X_scaled)

# 4. 查看结果
print("解释方差比:", pca.explained_variance_ratio_)
print("累积解释方差:", np.sum(pca.explained_variance_ratio_))
print("主成分形状:", X_pca.shape)
            """, language='python')
            
            if st.button("运行此代码", key="run_basic_pca"):
                # 执行示例代码
                pca_demo = PCA(n_components=2)
                X_pca_demo = pca_demo.fit_transform(X_scaled)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PC1解释方差", f"{pca_demo.explained_variance_ratio_[0]:.2%}")
                with col2:
                    st.metric("PC2解释方差", f"{pca_demo.explained_variance_ratio_[1]:.2%}")
                with col3:
                    st.metric("总解释方差", f"{np.sum(pca_demo.explained_variance_ratio_):.2%}")
        
        with code_tab2:
            st.markdown("##### 2. 手动实现PCA（理解原理）")
            st.code("""
import numpy as np

def manual_pca(X, n_components):
    '''
    手动实现PCA算法
    X: 输入数据，形状 (n_samples, n_features)
    n_components: 要保留的主成分数量
    '''
    # 1. 中心化数据（减去均值）
    X_centered = X - np.mean(X, axis=0)
    
    # 2. 计算协方差矩阵
    n = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    
    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. 按特征值降序排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. 选择前n个主成分
    components = eigenvectors[:, :n_components]
    
    # 6. 投影数据
    X_pca = X_centered @ components
    
    # 7. 计算解释方差比
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, components, explained_variance_ratio

# 使用示例
X_pca, components, explained_variance = manual_pca(X_scaled, 2)
            """, language='python')
            
            if st.button("运行手动实现", key="run_manual_pca"):
                # 执行手动实现
                def manual_pca_demo(X, n_components):
                    X_centered = X - np.mean(X, axis=0)
                    n = X_centered.shape[0]
                    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    components = eigenvectors[:, :n_components]
                    X_pca = X_centered @ components
                    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
                    return X_pca, components, explained_variance_ratio
                
                X_pca_manual, components_manual, explained_manual = manual_pca_demo(X_scaled, 2)
                
                st.success(f"手动实现成功！前两个主成分解释方差: {explained_manual[0]:.2%}, {explained_manual[1]:.2%}")
    
    with tab4:
        st.markdown('<div class="sub-header">📈 PCA案例应用</div>', unsafe_allow_html=True)
        
        case_study = st.selectbox(
            "选择案例",
            ["人脸识别中的降维", "文本数据降维", "高维传感器数据处理"],
            index=0
        )
        
        if case_study == "人脸识别中的降维":
            st.markdown("""
            ### 人脸识别案例：特征脸方法
            
            **问题背景:**
            人脸图像通常是高维数据（例如112×92=10304维），直接使用原始像素进行分类计算量大且容易过拟合。
            
            **PCA解决方案:**
            1. 使用PCA将人脸图像压缩到低维空间
            2. 在低维空间中进行分类
            3. 显著减少计算复杂度
            
            **关键优势:**
            - 降低维度：10304维 → 50-200维
            - 提取主要特征（特征脸）
            - 去除噪声和冗余信息
            
            **实现步骤:**
            1. 将所有人脸图像向量化
            2. 标准化数据
            3. 应用PCA提取主成分（特征脸）
            4. 在低维空间进行KNN分类
            """)
            
            # 模拟人脸数据降维效果
            if st.button("演示人脸数据降维效果"):
                # 使用Digits数据集模拟人脸数据
                from sklearn.datasets import fetch_olivetti_faces
                import matplotlib.pyplot as plt
                
                try:
                    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
                    X_faces = faces.data
                    y_faces = faces.target
                    
                    # 选择部分样本演示
                    n_samples = 100
                    X_faces_sample = X_faces[:n_samples]
                    y_faces_sample = y_faces[:n_samples]
                    
                    # 应用PCA
                    pca_faces = PCA(n_components=50)
                    X_faces_pca = pca_faces.fit_transform(X_faces_sample)
                    
                    # 显示结果
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**原始图像 (64×64=4096维)**")
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        for i, ax in enumerate(axes):
                            ax.imshow(X_faces_sample[i].reshape(64, 64), cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"人脸 {i+1}")
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**重建图像 (50维PCA)**")
                        X_reconstructed = pca_faces.inverse_transform(X_faces_pca)
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        for i, ax in enumerate(axes):
                            ax.imshow(X_reconstructed[i].reshape(64, 64), cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"重建 {i+1}")
                        st.pyplot(fig)
                    
                    # 解释方差
                    explained_ratio = np.sum(pca_faces.explained_variance_ratio_)
                    st.success(f"使用50个主成分保留了 {explained_ratio:.2%} 的信息")
                    
                except:
                    st.info("由于网络限制，无法加载人脸数据集。这里演示了PCA的核心思想。")
        
        elif case_study == "文本数据降维":
            st.markdown("""
            ### 文本数据案例：文档主题提取
            
            **问题背景:**
            文本数据经过TF-IDF或词袋模型处理后通常是高维稀疏矩阵（数千到数万维）。
            
            **PCA解决方案:**
            1. 将文档-词矩阵降维到主题空间
            2. 每个主成分代表一个潜在主题
            3. 在低维空间进行文档聚类或分类
            
            **关键优势:**
            - 处理高维稀疏数据
            - 发现潜在语义结构
            - 提高后续任务性能
            """)

# t-SNE页面
def tsne_page():
    st.markdown('<div class="main-header">🌀 t-SNE 可视化</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ t-SNE参数设置")
        
        dataset_option = st.selectbox(
            "数据集",
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
            index=0 if 'current_dataset' not in st.session_state else 
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"].index(st.session_state.current_dataset)
        )
        
        X, y, feature_names, target_names, data_type = load_data(dataset_option)
        
        # t-SNE参数
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("学习率", 10, 1000, 200)
        n_iter = st.slider("迭代次数", 250, 2000, 1000)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # 执行t-SNE
    if st.sidebar.button("运行t-SNE", type="primary"):
        with st.spinner("t-SNE正在运行中..."):
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )
            X_tsne = tsne.fit_transform(X_scaled)
            
            # 可视化结果
            fig = px.scatter(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                color=y.astype(str) if len(np.unique(y)) < 10 else None,
                title=f"t-SNE可视化 (Perplexity={perplexity})",
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_traces(
                marker=dict(size=8, opacity=0.7),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 参数解释
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **参数解释:**
            - **Perplexity**: 平衡局部和全局结构，通常建议值在5-50之间
            - **学习率**: 控制优化步长，太高可能导致不稳定
            - **迭代次数**: 优化迭代次数，确保收敛
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # t-SNE理论解释
    st.markdown('<div class="sub-header">📖 t-SNE理论基础</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **t-SNE核心思想:**
        
        1. **保持概率分布**
           - 在高维空间计算样本对的相似度
           - 在低维空间保持相同的相似度分布
        
        2. **t-分布**
           - 低维空间使用t分布计算相似度
           - 解决"拥挤问题"
        
        3. **KL散度最小化**
           - 最小化高低维分布的KL散度
           - 梯度下降优化
        """)
    
    with col2:
        st.latex(r"""
        \begin{aligned}
        &p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} \\
        &q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}} \\
        &C = KL(P\|Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
        \end{aligned}
        """)

# UMAP页面
def umap_page():
    st.markdown('<div class="main-header">🌌 UMAP 降维</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ UMAP参数设置")
        
        dataset_option = st.selectbox(
            "数据集",
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
            index=0
        )
        
        X, y, feature_names, target_names, data_type = load_data(dataset_option)
        
        # UMAP参数
        n_neighbors = st.slider("邻居数量", 2, 100, 15)
        min_dist = st.slider("最小距离", 0.0, 1.0, 0.1)
        n_components = st.slider("输出维度", 2, 3, 2)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # 执行UMAP
    if st.sidebar.button("运行UMAP", type="primary"):
        with st.spinner("UMAP正在运行中..."):
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=42
            )
            X_umap = reducer.fit_transform(X_scaled)
            
            # 可视化
            if n_components == 2:
                fig = px.scatter(
                    x=X_umap[:, 0],
                    y=X_umap[:, 1],
                    color=y.astype(str) if len(np.unique(y)) < 10 else None,
                    title=f"UMAP可视化 (n_neighbors={n_neighbors})",
                    labels={"x": "UMAP 1", "y": "UMAP 2"},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
            else:
                fig = go.Figure(data=[go.Scatter3d(
                    x=X_umap[:, 0],
                    y=X_umap[:, 1],
                    z=X_umap[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=y if len(np.unique(y)) < 10 else 'blue',
                        colorscale='Viridis',
                        opacity=0.7
                    )
                )])
                
                fig.update_layout(
                    title=f"UMAP 3D可视化",
                    scene=dict(
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        zaxis_title="UMAP 3"
                    )
                )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # UMAP优势
    st.markdown('<div class="sub-header">✨ UMAP优势</div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    advantages = [
        ("🚀 速度快", "比t-SNE更快，适合大数据集"),
        ("🌐 保持全局结构", "同时保持局部和全局结构"),
        ("🔧 参数少", "主要参数只有邻居数量和最小距离"),
        ("📈 可扩展性", "支持增量学习和新样本投影"),
        ("🎯 稳定性", "对超参数不敏感，结果稳定"),
        ("💾 内存效率", "内存消耗相对较低")
    ]
    
    for col, (title, desc) in zip(cols, advantages):
        with col:
            st.markdown(f"""
            <div class="card">
            <h4>{title}</h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# LDA页面
def lda_page():
    st.markdown('<div class="main-header">🎯 线性判别分析 (LDA)</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ LDA参数设置")
        
        dataset_option = st.selectbox(
            "数据集",
            ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
            index=0
        )
        
        X, y, feature_names, target_names, data_type = load_data(dataset_option)
        
        # LDA参数
        n_components = st.slider("成分数量", 1, min(9, len(np.unique(y))-1), 2)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # 执行LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # 可视化
    if n_components >= 2:
        fig = px.scatter(
            x=X_lda[:, 0],
            y=X_lda[:, 1],
            color=y.astype(str),
            title="LDA降维结果",
            labels={"x": f"LD1 (解释方差: {lda.explained_variance_ratio_[0]:.2%})",
                   "y": f"LD2 (解释方差: {lda.explained_variance_ratio_[1]:.2%})"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # LDA与PCA对比
    st.markdown('<div class="sub-header">⚖️ LDA vs PCA</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **LDA (监督学习)**
        
        **目标:**
        - 最大化类间距离
        - 最小化类内距离
        
        **数学目标:**
        """)
        st.latex(r"J(w) = \frac{w^T S_B w}{w^T S_W w}")
        st.markdown("""
        **适用场景:**
        - 分类问题特征提取
        - 类别信息已知
        - 提升分类器性能
        """)
    
    with col2:
        st.markdown("""
        **PCA (无监督学习)**
        
        **目标:**
        - 最大化总体方差
        - 特征去相关
        
        **数学目标:**
        """)
        st.latex(r"\max_w w^T \Sigma w \quad \text{s.t.} \quad w^T w = 1")
        st.markdown("""
        **适用场景:**
        - 数据探索
        - 特征提取
        - 数据压缩
        - 噪声过滤
        """)
    
    # 分类性能对比
    st.markdown('<div class="section-header">📊 分类性能对比</div>', unsafe_allow_html=True)
    
    if st.button("比较PCA和LDA的分类效果"):
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # PCA降维
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # LDA降维
        lda_comp = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_lda = lda_comp.fit_transform(X_train, y_train)
        X_test_lda = lda_comp.transform(X_test)
        
        # 训练分类器
        knn = KNeighborsClassifier(n_neighbors=3)
        
        # 原始数据
        knn.fit(X_train, y_train)
        y_pred_orig = knn.predict(X_test)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        
        # PCA降维后
        knn.fit(X_train_pca, y_train)
        y_pred_pca = knn.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, y_pred_pca)
        
        # LDA降维后
        knn.fit(X_train_lda, y_train)
        y_pred_lda = knn.predict(X_test_lda)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        
        # 显示结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始数据准确率", f"{acc_orig:.2%}")
        with col2:
            st.metric("PCA降维后准确率", f"{acc_pca:.2%}", 
                     f"{(acc_pca-acc_orig):+.2%}")
        with col3:
            st.metric("LDA降维后准确率", f"{acc_lda:.2%}",
                     f"{(acc_lda-acc_orig):+.2%}")

# 对比分析页面
def comparison_page():
    st.markdown('<div class="main-header">📊 降维方法对比</div>', unsafe_allow_html=True)
    
    # 数据集选择
    dataset_option = st.selectbox(
        "选择数据集",
        ["鸢尾花数据集 (Iris)", "手写数字数据集 (Digits)", "葡萄酒数据集 (Wine)", "随机生成数据"],
        index=0
    )
    
    X, y, feature_names, target_names, data_type = load_data(dataset_option)
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行不同降维方法
    methods = ["PCA", "t-SNE", "UMAP", "LDA"]
    results = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        results["PCA"] = X_pca
        
        fig1 = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=y.astype(str),
            title="PCA",
            labels={"x": "PC1", "y": "PC2"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        st.info(f"解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    with col2:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        results["t-SNE"] = X_tsne
        
        fig2 = px.scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            color=y.astype(str),
            title="t-SNE",
            labels={"x": "t-SNE1", "y": "t-SNE2"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        results["UMAP"] = X_umap
        
        fig3 = px.scatter(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            color=y.astype(str),
            title="UMAP",
            labels={"x": "UMAP1", "y": "UMAP2"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # LDA (仅当有类别标签且类别数>1时)
        if len(np.unique(y)) > 1:
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(X_scaled, y)
            results["LDA"] = X_lda
            
            fig4 = px.scatter(
                x=X_lda[:, 0],
                y=X_lda[:, 1],
                color=y.astype(str),
                title="LDA",
                labels={"x": "LD1", "y": "LD2"},
                color_discrete_sequence=px.colors.qualitative.Pastel1
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    # 方法对比表格
    st.markdown('<div class="sub-header">📋 方法特性对比</div>', unsafe_allow_html=True)
    
    comparison_data = {
        "特性": ["监督/无监督", "线性/非线性", "保持全局结构", "保持局部结构", "计算复杂度", "适合大样本", "参数敏感性"],
        "PCA": ["无监督", "线性", "✓", "✗", "低", "✓", "低"],
        "t-SNE": ["无监督", "非线性", "✗", "✓", "高", "✗", "高"],
        "UMAP": ["无监督", "非线性", "✓", "✓", "中", "✓", "中"],
        "LDA": ["监督", "线性", "✓", "✗", "低", "✓", "低"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

# 问答页面
def qa_page():
    st.markdown('<div class="main-header">❓ 常见问题解答</div>', unsafe_allow_html=True)
    
    questions = [
        {
            "question": "PCA为什么要对数据进行标准化？",
            "answer": """
            标准化是PCA的重要预处理步骤，原因包括：
            
            1. **消除量纲影响**：不同特征可能具有不同的量纲和取值范围，标准化确保所有特征平等对待。
            2. **防止方差主导**：方差大的特征会主导主成分方向，这可能不是真实的数据结构。
            3. **数值稳定性**：提高计算的数值稳定性。
            4. **数学要求**：PCA基于协方差矩阵，标准化后协方差矩阵等于相关矩阵。
            
            **示例**：如果特征1的范围是0-100，特征2的范围是0-1，未标准化时PCA会过度关注特征1。
            """,
            "category": "PCA"
        },
        {
            "question": "t-SNE为什么不适合大样本数据集？",
            "answer": """
            t-SNE的主要限制包括：
            
            1. **计算复杂度高**：时间复杂度为O(N²)，内存消耗为O(N²)，N为样本数。
            2. **内存消耗大**：需要存储N×N的相似度矩阵。
            3. **计算时间长**：大规模数据集可能需要数小时甚至数天。
            
            **解决方案**：
            - 使用UMAP作为替代（复杂度O(N)）
            - 先使用PCA降维到50维左右，再应用t-SNE
            - 使用随机子采样
            """,
            "category": "t-SNE"
        },
        {
            "question": "降维后的数据还能还原回原始空间吗？",
            "answer": """
            这取决于降维方法：
            
            **可逆的降维**：
            - **PCA**：可以近似还原（有信息损失）
            ```python
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            X_reconstructed = pca.inverse_transform(X_reduced)
            ```
            
            **不可逆的降维**：
            - **t-SNE**：不能还原，因为是复杂的非线性映射
            - **UMAP**：理论可逆但实现复杂
            
            **还原质量取决于**：
            1. 保留的主成分数量
            2. 原始数据的结构
            3. 降维方法的选择
            """,
            "category": "通用"
        },
        {
            "question": "如何选择主成分的数量？",
            "answer": """
            有几种常用方法：
            
            1. **累积方差解释率**：通常选择累积解释方差≥95%的最小k值
            ```python
            pca = PCA()
            pca.fit(X)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            k = np.argmax(cumsum >= 0.95) + 1
            ```
            
            2. **肘部法则**：绘制特征值，选择"肘部"点
            3. **交叉验证**：基于下游任务性能选择
            
            **经验法则**：
            - 数据可视化：2-3个成分
            - 特征提取：保留95%方差
            - 去噪：丢弃特征值接近0的成分
            """,
            "category": "PCA"
        },
        {
            "question": "PCA和LDA的主要区别是什么？",
            "answer": """
            **核心区别**：
            
            | 特性 | PCA (无监督) | LDA (监督) |
            |------|-------------|-----------|
            | **目标** | 最大化方差 | 最大化类间分离度 |
            | **使用标签** | 不使用 | 使用 |
            | **数学目标** | max wᵀΣw | max (wᵀS_B w)/(wᵀS_W w) |
            | **输出维度** | ≤min(n_features, n_samples) | ≤min(n_features, n_classes-1) |
            | **适用场景** | 数据探索、压缩 | 分类特征提取 |
            
            **简单记忆**：
            - PCA：找数据最分散的方向
            - LDA：找类别最分离的方向
            """,
            "category": "对比"
        }
    ]
    
    # 按类别分组显示
    categories = sorted(set(q["category"] for q in questions))
    
    for category in categories:
        st.markdown(f'<div class="section-header">{category}</div>', unsafe_allow_html=True)
        
        category_questions = [q for q in questions if q["category"] == category]
        
        for i, qa in enumerate(category_questions):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.markdown(qa["answer"])

# 主应用逻辑
def main():
    # 侧边栏导航
    with st.sidebar:
        st.markdown("# 📚 导航菜单")
        
        page = st.radio(
            "选择学习模块",
            ["🏠 首页", 
             "📊 PCA主成分分析", 
             "🌀 t-SNE可视化", 
             "🌌 UMAP降维", 
             "🎯 LDA判别分析",
             "📋 方法对比",
             "❓ 问答专区"]
        )
        
        st.markdown("---")
        st.markdown("## 📖 学习资源")
        st.markdown("""
        - [scikit-learn文档](https://scikit-learn.org/)
        - [UMAP官方文档](https://umap-learn.readthedocs.io/)
        - [交互式线性代数](https://textbooks.math.gatech.edu/ila/)
        """)
        
        st.markdown("---")
        st.markdown("## 🛠️ 工具信息")
        st.markdown(f"""
        - Streamlit版本: {st.__version__}
        - 数据维度: 自动检测
        - 计算模式: 本地执行
        """)
    
    # 页面路由
    if page == "🏠 首页":
        home_page()
    elif page == "📊 PCA主成分分析":
        pca_page()
    elif page == "🌀 t-SNE可视化":
        tsne_page()
    elif page == "🌌 UMAP降维":
        umap_page()
    elif page == "🎯 LDA判别分析":
        lda_page()
    elif page == "📋 方法对比":
        comparison_page()
    elif page == "❓ 问答专区":
        qa_page()

if __name__ == "__main__":
    main()
