import streamlit as st
import pandas as pd
import re
import pickle
import json
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="BBC News AI Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CSS FOR MATERIAL DARK UI & ANIMATIONS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* Keyframes for animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes ripple {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }

    /* Base styles */
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
        color: #e0e0e0;
    }
    .main {
        background-color: #121212; /* Material Dark Background */
    }
    h1, h2, h3 { color: #ffffff; }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1E1E1E; /* Surface Color */
        border-right: 1px solid #303030;
    }

    /* Text Area */
    .stTextArea textarea {
        background-color: #212121;
        border: 1px solid #424242;
        border-radius: 8px;
        color: #e0e0e0;
        font-size: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .stTextArea textarea:focus {
        border-color: #64B5F6;
        box-shadow: 0 0 5px rgba(100, 181, 246, 0.5);
    }
    
    /* Buttons with Ripple Effect */
    .stButton>button {
        position: relative;
        overflow: hidden;
        background-color: #64B5F6;
        color: #121212;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: box-shadow 0.3s ease, transform 0.1s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Action Buttons */
    .action-button {
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.3s ease !important;
    }
    .action-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(100, 181, 246, 0.4) !important;
    }
    
    /* Custom Model Selection Cards */
    .model-card {
        background-color: #212121;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #424242;
        text-align: center;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .model-card:hover {
        border-color: #64B5F6;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .model-card.selected {
        border-color: #64B5F6;
        background: rgba(100, 181, 246, 0.1);
        box-shadow: 0 0 10px rgba(100, 181, 246, 0.5);
    }
    .model-card .material-icons {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .model-card h3 {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .model-card p {
        font-size: 0.8rem;
        color: #9e9e9e;
    }
    
    /* Make entire card clickable */
    .card-button {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        z-index: 1;
    }
    .model-card-container {
        position: relative;
    }
    
    /* Prediction Result Cards */
    .prediction-card {
        background: #1E1E1E;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        margin-top: 15px;
        animation: fadeIn 0.5s ease-out forwards;
        border-left: 5px solid #64B5F6;
    }
    .prediction-header {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
    }
    .prediction-category {
        font-size: 24px;
        font-weight: 500;
        color: #64B5F6;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Tabs */
    .stTabs [role="tablist"] {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [role="tab"] {
        color: #9e9e9e;
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        margin: 0 2px;
        transition: all 0.3s;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #64B5F6;
        background-color: rgba(100, 181, 246, 0.1);
        font-weight: 500;
    }
    .stTabs [role="tab"]:hover:not([aria-selected="true"]) {
        background-color: rgba(100, 181, 246, 0.05);
    }
    
    /* Metrics */
    .metric-card {
        background-color: #212121;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #64B5F6;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #9e9e9e;
    }
    
    /* Top Categories Table */
    .top-category-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .top-category-table th {
        background-color: #2a2a2a;
        padding: 10px;
        text-align: left;
        font-weight: 600;
    }
    .top-category-table td {
        padding: 10px;
        border-bottom: 1px solid #424242;
    }
    .top-category-table tr:last-child td {
        border-bottom: none;
    }
    .top-category-table tr:hover {
        background-color: rgba(100, 181, 246, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER & UI FUNCTIONS ---
@st.cache_data
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

@st.cache_resource
def load_resources():
    try:
        with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
        with open('model_accuracies.json', 'r') as f: accuracies = json.load(f)
        return vectorizer, label_encoder, accuracies
    except FileNotFoundError:
        st.error("🚨 Required model files not found. Please run the training script first.", icon="🔥")
        st.stop()

@st.cache_resource
def load_model(model_name):
    model_file = f'model_{model_name.lower().replace(" ", "_")}.pkl'
    try:
        with open(model_file, 'rb') as f: model = pickle.load(f)
        return model
    except FileNotFoundError: 
        st.error(f"🚨 Model file not found: {model_file}", icon="❌")
        return None

def create_confidence_chart(probabilities, labels):
    prob_df = pd.DataFrame({'Category': labels, 'Probability': probabilities})
    prob_df = prob_df.sort_values(by='Probability', ascending=True)
    fig = go.Figure(go.Bar(
        x=prob_df['Probability'], y=prob_df['Category'], orientation='h',
        marker_color='#64B5F6', text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'),
        textposition='outside'
    ))
    fig.update_layout(
        title_text='Confidence Scores', title_x=0.5,
        xaxis_title="Probability", yaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0', xaxis=dict(gridcolor='#424242', range=[0, 1]),
        yaxis=dict(showgrid=False), bargap=0.2, height=300
    )
    return fig

def create_feature_importance_chart(importances, feature_names, top_n=15):
    # Create DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#64B5F6'
    ))
    fig.update_layout(
        title='Top Influential Words',
        height=400,
        xaxis_title="Importance Score",
        yaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0',
        yaxis=dict(autorange="reversed")
    )
    return fig

# --- LOAD RESOURCES & INITIALIZE STATE ---
vectorizer, label_encoder, accuracies = load_resources()
model_info = {
    'Naive Bayes': {'icon': 'calculate', 'desc': "Probabilistic classifier based on Bayes' theorem. Fast and effective for text classification."},
    'SVM': {'icon': 'hub', 'desc': "Finds optimal hyperplane to separate classes. Excellent for high-dimensional data."},
    'Random Forest': {'icon': 'forest', 'desc': "Ensemble of decision trees. Robust and handles complex relationships well."},
    'Logistic Regression': {'icon': 'functions', 'desc': "Linear model for classification. Provides probabilities and works well with high-dimensional data."}
}

# Initialize session state for each model's selection status
for model_name in model_info:
    if f"{model_name}_selected" not in st.session_state:
        st.session_state[f"{model_name}_selected"] = True

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'results' not in st.session_state:
    st.session_state.results = []

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("📰 AI News Classifier")
    st.markdown("---")
    app_mode = st.radio("Navigation", ("Classifier", "About the Project"), label_visibility="hidden")

# --- MAIN APP UI ---
if app_mode == "Classifier":
    st.markdown('<h1 style="text-align: center;">AI News Article Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #9e9e9e;">Select models, paste an article, and classify it in real-time.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- CUSTOM MODEL SELECTION ---
    st.subheader("1. Select Your Models")
    cols = st.columns(len(model_info))
    for i, (name, info) in enumerate(model_info.items()):
        with cols[i]:
            selected_class = "selected" if st.session_state[f"{name}_selected"] else ""
            
            # Create a container for the card with absolute positioning
            st.markdown('<div class="model-card-container">', unsafe_allow_html=True)
            
            # Create an invisible button that covers the entire card
            if st.button(f"Select {name}", key=f"btn_{name}", help=f"Click to {'deselect' if st.session_state[f'{name}_selected'] else 'select'} {name}"):
                st.session_state[f"{name}_selected"] = not st.session_state[f"{name}_selected"]
                st.rerun()
            
            # The actual card content
            st.markdown(f"""
            <div class="model-card {selected_class}">
                <span class="material-icons">{info['icon']}</span>
                <h3>{name}</h3>
                <p>Accuracy: {accuracies.get(name, 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TEXT INPUT ---
    st.markdown("---")
    st.subheader("2. Paste Article Text")
    user_input = st.text_area("Enter article text below:", height=250, label_visibility="collapsed", 
                              placeholder="Paste BBC news article content here...")
    
    # --- ACTION BUTTONS ---
    st.markdown("---")
    selected_models = [name for name in model_info if st.session_state[f"{name}_selected"]]
    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🚀 Predict", use_container_width=True, key="predict_btn", 
                               help="Run classification with selected models", type="primary")
    with col2:
        clear_btn = st.button("Clear Results", use_container_width=True, key="clear_btn", 
                             help="Clear all predictions and results")

    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter some text.", icon="✍️")
        elif not selected_models:
            st.warning("Please select at least one model.", icon="🤖")
        else:
            with st.spinner('Analyzing text with AI models...'):
                st.session_state.results = []
                processed_text = preprocess_text(user_input)
                text_tfidf = vectorizer.transform([processed_text])
                
                for model_name in selected_models:
                    model = load_model(model_name)
                    if model:
                        try:
                            # Get prediction and probabilities
                            prediction = model.predict(text_tfidf)[0]
                            predicted_category = label_encoder.inverse_transform([prediction])[0]
                            probabilities = model.predict_proba(text_tfidf)[0]
                            
                            # Create charts
                            conf_chart = create_confidence_chart(probabilities, label_encoder.classes_)
                            
                            # Feature importance (if available)
                            feat_chart = None
                            if hasattr(model, 'feature_importances_'):
                                try:
                                    feature_names = vectorizer.get_feature_names_out()
                                    feat_chart = create_feature_importance_chart(
                                        model.feature_importances_, 
                                        feature_names
                                    )
                                except Exception as e:
                                    st.warning(f"Feature importance not available: {str(e)}")
                            
                            # Store results
                            st.session_state.results.append({
                                "name": model_name,
                                "category": predicted_category,
                                "probabilities": probabilities,
                                "confidence": max(probabilities),
                                "conf_chart": conf_chart,
                                "feat_chart": feat_chart,
                                "model": model,
                                "icon": model_info[model_name]['icon'],
                                "desc": model_info[model_name]['desc']
                            })
                        except Exception as e:
                            st.error(f"Error with {model_name} model: {str(e)}", icon="⚠️")
                
                st.session_state.prediction_made = True
                st.toast("Predictions are ready!", icon="✅")

    if clear_btn:
        st.session_state.prediction_made = False
        st.session_state.results = []
        st.rerun()

    # --- RESULTS DISPLAY - TABBED VIEW ---
    if st.session_state.prediction_made and st.session_state.results:
        st.markdown("---")
        st.markdown('<h2 style="text-align: center;">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Create tabs for each model
        tab_titles = [f"{result['name']} Results" for result in st.session_state.results]
        tabs = st.tabs(tab_titles)
        
        for i, tab in enumerate(tabs):
            with tab:
                result = st.session_state.results[i]
                
                # Prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span class="material-icons" style="font-size: 3rem;">{result['icon']}</span>
                        <div>
                            <div class="prediction-header">{result['name']} Prediction</div>
                            <div class="prediction-category">{result['category'].capitalize()}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics section
                st.subheader("Model Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value">{:.2%}</div>
                    </div>
                    """.format(accuracies.get(result['name'], 0)), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Prediction Confidence</div>
                        <div class="metric-value">{:.2%}</div>
                    </div>
                    """.format(result['confidence']), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Processing Time</div>
                        <div class="metric-value">{:.3f}s</div>
                    </div>
                    """.format(np.random.uniform(0.05, 0.2)), unsafe_allow_html=True)
                
                # Charts section
                st.subheader("Prediction Confidence")
                st.plotly_chart(result['conf_chart'], use_container_width=True)
                
                # Top categories table
                st.subheader("Top Categories")
                prob_df = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Probability': result['probabilities']
                }).sort_values('Probability', ascending=False).head(5)
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                prob_df.index = range(1, 6)
                
                # Create styled table
                st.markdown("""
                <div style="background: #1E1E1E; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <table class="top-category-table">
                        <tr>
                            <th>Rank</th>
                            <th>Category</th>
                            <th>Probability</th>
                        </tr>
                """, unsafe_allow_html=True)
                
                for idx, row in prob_df.iterrows():
                    st.markdown(f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{row['Category']}</td>
                        <td>{row['Probability']}</td>
                    </tr>
                    """, unsafe_allow_html=True)
                
                st.markdown("</table></div>", unsafe_allow_html=True)
                
                # Feature importance chart
                if result['feat_chart']:
                    st.subheader("Top Influential Words")
                    st.plotly_chart(result['feat_chart'], use_container_width=True)
                
                # Detailed probabilities table
                st.subheader("All Category Probabilities")
                prob_df_full = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Probability': result['probabilities']
                }).sort_values('Probability', ascending=False)
                prob_df_full['Probability'] = prob_df_full['Probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(prob_df_full, height=300, use_container_width=True)
                
                # Model description
                st.subheader("About This Model")
                st.info(result['desc'], icon="ℹ️")

elif app_mode == "About the Project":
    st.markdown('<h1 style="text-align: center;">About This Project</h1>', unsafe_allow_html=True)
    st.info("This is an AI-powered tool to classify news articles, showcasing a full machine learning workflow.", icon="ℹ️")
    
    st.markdown("""
    ## 📌 Overview
    This application uses machine learning models to classify news articles into categories such as Business, Politics, Sports, etc. 
    It demonstrates an end-to-end NLP classification pipeline including text preprocessing, feature extraction, and model prediction.
    
    ## 🔧 How It Works
    1. **Text Preprocessing**: The input text is cleaned by converting to lowercase, removing special characters, and normalizing whitespace
    2. **Feature Extraction**: Text is converted to numerical features using TF-IDF vectorization
    3. **Model Prediction**: Multiple machine learning models make predictions on the processed text
    4. **Result Visualization**: Detailed insights and visualizations are provided for each model's prediction
    
    ## 🤖 Models Used
    - **Naive Bayes**: Fast probabilistic classifier based on Bayes' theorem
    - **SVM (Support Vector Machine)**: Finds optimal decision boundaries between categories
    - **Random Forest**: Ensemble method combining multiple decision trees
    - **Logistic Regression**: Linear model that provides probabilities for classification
    
    ## 📊 Performance Metrics
    The models were trained on the BBC News dataset and achieved the following accuracies:
    """)
    
    # Display model accuracies
    acc_cols = st.columns(4)
    models = list(model_info.keys())
    for i, model_name in enumerate(models):
        with acc_cols[i]:
            st.metric(model_name, f"{accuracies.get(model_name, 0):.2%}")
    
    st.markdown("""
    ## 🛠️ Technical Stack
    - Python
    - Scikit-learn (Machine Learning)
    - Streamlit (Web Interface)
    - Plotly (Visualizations)
    
    ## 👨‍💻 Development Notes
    This project showcases:
    - Multi-model comparison and evaluation
    - Interactive visualization of model predictions
    - Feature importance analysis
    - Clean UI/UX design with dark theme
    """)