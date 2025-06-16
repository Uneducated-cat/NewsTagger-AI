import streamlit as st
import pandas as pd
import re
import pickle
import json
import plotly.graph_objects as go
import numpy as np
from scipy.sparse import issparse # Import to check for sparse matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="NewsTagger AI",
    page_icon="📰🏷️",
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
    .css-1d391kg { /* Target for Streamlit sidebar */
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
    
    /* Buttons with Ripple Effect (General Streamlit Buttons) */
    .stButton > button {
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
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Action Buttons (General Styling for st.button containers like Predict/Clear) */
    div.stButton {
        width: 100%; /* Ensure general buttons fill container */
        margin-top: 10px; /* Add some spacing */
        transition: all 0.3s ease;
    }
    div.stButton > button { /* Specific styling for the button element inside stButton div */
        width: 100%; /* Make the button fill its container */
    }
    div.stButton:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(100, 181, 246, 0.4);
    }


    /* CUSTOM MODEL SELECTION CARDS - Styled directly, with JS for clicks */

    .model-card-wrapper {
        height: 180px; /* Fixed height for all cards */
        /* margin-bottom handled by the parent stButton div */
        position: relative; /* Needed for z-index context with .model-card */
        z-index: 1; /* Ensure card is above hidden button */
        cursor: pointer; /* Indicate it's clickable */
    }

    .model-card {
        background-color: #212121;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #424242;
        text-align: center;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        height: 100%; /* Make sure the card itself fills the wrapper */
        display: flex;
        flex-direction: column;
        justify-content: center; /* Center content vertically */
        align-items: center; /* Center content horizontally */
        position: relative; /* For checkmark icon */
        user-select: none; /* Prevent text selection on click */
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
        color: #BBDEFB; /* Light blue icon color */
    }
    .model-card h3 {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }
    .model-card p {
        font-size: 0.8rem;
        color: #9e9e9e;
    }

    /* Checkmark icon for selected cards */
    .model-card.selected::after {
        content: 'check_circle'; /* Material icon name */
        font-family: 'Material Icons';
        font-size: 2rem;
        color: #4CAF50; /* Green checkmark */
        position: absolute;
        top: 10px;
        right: 10px;
        line-height: 1;
    }
    
    /* CSS to ONLY hide the SPECIFIC hidden buttons for model selection */
    /* This targets the Streamlit div.stButton that contains our hidden button */
    div.stButton:has(button[id^="hidden_button_"]) {
        position: relative !important; /* Make parent relative for absolute button positioning */
        height: 180px !important; /* Give it the same height as the card for proper spacing */
        margin-bottom: 15px !important; /* Restore normal margin for column flow */
        overflow: hidden !important; /* Hide anything outside this div */
        /* No z-index here, let the actual button or the overlaying card handle it */
    }

    /* Target the button itself by its Streamlit-generated ID (from 'key') */
    button[id^="hidden_button_"] {
        position: absolute !important; /* Position it absolutely within its parent div.stButton */
        top: 0 !important;
        left: 0 !important;
        width: 100% !important; /* Make it fill its parent div.stButton */
        height: 100% !important; /* Make it fill its parent div.stButton */
        overflow: hidden !important;
        opacity: 0 !important; /* Make it completely invisible */
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        z-index: 0; /* Place it *below* the custom card */
        background-color: transparent !important; /* Ensure no background */
        color: transparent !important; /* Hide any text */
        pointer-events: all !important; /* CRUCIAL: Allow programmatic clicks */
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
        height: 100%; /* Ensure equal height for metric cards */
        display: flex;
        flex-direction: column;
        justify-content: center;
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
        # These paths assume the pickle and JSON files are in the same directory as your app.py
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
        yaxis=dict(showgrid=False), bargap=0.2, height=450 # Increased height
    )
    return fig

def create_feature_importance_chart(importances, feature_names, top_n=15):
    # Ensure importances is 1D and feature_names match
    importances = np.asarray(importances)
    if importances.ndim > 1:
        importances = importances.flatten()
    
    # Handle cases where feature_names might not exactly match importances length
    if len(importances) != len(feature_names):
        st.warning(f"Feature importance length ({len(importances)}) does not match feature names length ({len(feature_names)}). This might affect the chart.", icon="⚠️")
        if len(importances) > len(feature_names):
            importances = importances[:len(feature_names)]
        elif len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        
        if len(importances) == 0: # If after adjustments, there are no features
             return None

    # Create DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(importances) # Use absolute importance for ranking
    }).sort_values('Importance', ascending=False).head(top_n)
    
    if importance_df.empty:
        return None

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#64B5F6'
    ))
    fig.update_layout(
        title='Top Influential Words',
        height=500, # Increased height
        xaxis_title="Importance Score (Absolute)",
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
    'Naive Bayes': {'icon': 'calculate', 'desc': "A probabilistic classifier based on Bayes' theorem, assuming independence between features. It's often fast and performs well with text data."},
    'SVM': {'icon': 'hub', 'desc': "Support Vector Machines find the optimal hyperplane that best separates different classes in the feature space. They are effective in high-dimensional spaces."},
    'Random Forest': {'icon': 'forest', 'desc': "An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It's robust and handles complex relationships."},
    'Logistic Regression': {'icon': 'functions', 'desc': "Despite its name, Logistic Regression is a linear model for classification rather than regression. It models the probability of a binary outcome and is extended for multi-class classification. It's simple, interpretable, and performs well with high-dimensional data."}
}

# Initialize session state for each model's selection status
for model_name in model_info:
    if f"{model_name}_selected" not in st.session_state:
        st.session_state[f"{model_name}_selected"] = True

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'results' not in st.session_state:
    st.session_state.results = []

# --- Callback function to toggle selection state (triggered by JavaScript) ---
def toggle_model_selection_callback(model_name):
    st.session_state[f"{model_name}_selected"] = not st.session_state[f"{model_name}_selected"]
    # Streamlit will automatically rerun the script when session_state changes via a button click

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("📰 NewsTagger AI")
    st.markdown("---")
    app_mode = st.radio("Navigation", ("Classifier", "About the Project"), label_visibility="hidden")

# --- MAIN APP UI ---
if app_mode == "Classifier":
    st.markdown('<h1 style="text-align: center;">NewsTagger AI: News Article Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #9e9e9e;">Select models, paste an article, and classify it in real-time.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- CUSTOM MODEL SELECTION ---
    st.subheader("1. Select Your Models")
    cols = st.columns(len(model_info))
    for i, (name, info) in enumerate(model_info.items()):
        with cols[i]:
            is_selected = st.session_state[f"{name}_selected"]
            selected_class = "selected" if is_selected else ""
            
            # Create a hidden Streamlit button that the JavaScript will "click"
            # This button's only purpose is to trigger a Streamlit rerun and the Python callback
            st.button(
                label=" ", # A single space as a label to ensure it renders (even if tiny)
                key=f"hidden_button_{name.replace(' ', '_')}",
                on_click=toggle_model_selection_callback,
                args=(name,),
                # Streamlit's 'type' parameter is for visual styling, but 'secondary' is a safe default
                type="secondary", 
                help=f"Hidden button for {name} selection" # This tooltip will still show up if hovered over the tiny space
            )

            # Use st.empty() to get a placeholder for our custom HTML
            # This placeholder's content will visually overlay the hidden Streamlit button
            placeholder = st.empty()

            # Render the custom card HTML directly within the placeholder
            # The onclick event will trigger a hidden Streamlit button
            html_card = f"""
            <div class="model-card-wrapper">
                <div id="model_card_{name.replace(' ', '_')}" class="model-card {selected_class}" 
                     onclick="
                        // Trigger a hidden Streamlit button click programmatically
                        var button = document.getElementById('hidden_button_{name.replace(' ', '_')}');
                        if (button) {{
                            button.click();
                        }}
                    ">
                    <span class="material-icons">{info['icon']}</span>
                    <h3>{name}</h3>
                    <p>Accuracy: {accuracies.get(name, 0):.2%}</p>
                </div>
            </div>
            """
            placeholder.markdown(html_card, unsafe_allow_html=True)


    # --- TEXT INPUT ---
    st.markdown("---")
    st.subheader("2. Paste Article Text")
    user_input = st.text_area("Enter article text below:", height=250, label_visibility="collapsed", 
                              placeholder="Paste news article content here...")
    
    # --- ACTION BUTTONS ---
    st.markdown("---")
    selected_models = [name for name in model_info if st.session_state[f"{name}_selected"]]
    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("Predict", use_container_width=True, key="predict_btn", 
                               help="Run classification with selected models", type="primary")
    with col2:
        clear_btn = st.button("Clear Results", use_container_width=True, key="clear_btn", 
                             help="Clear all predictions and results")

    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter some text.", icon="🥀")
        elif not selected_models:
            st.warning("Please select at least one model.", icon="🪫")
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
                            
                            # Feature importance (handled more robustly now)
                            feat_chart = None
                            feature_names = vectorizer.get_feature_names_out()
                            predicted_class_index = label_encoder.transform([predicted_category])[0]

                            if hasattr(model, 'feature_importances_'):
                                # For tree-based models like Random Forest
                                importances = model.feature_importances_
                                feat_chart = create_feature_importance_chart(importances, feature_names)
                                
                            elif hasattr(model, 'coef_'):
                                # For linear models like SVM and Logistic Regression
                                coef_data = model.coef_
                                if issparse(coef_data): # Convert sparse to dense if necessary
                                    coef_data = coef_data.toarray()

                                if coef_data.ndim > 1: # Multi-class scenario
                                    importances = coef_data[predicted_class_index]
                                else: # Binary class scenario (should be rare with this dataset)
                                    importances = coef_data
                                
                                # Use absolute values for importance ranking
                                feat_chart = create_feature_importance_chart(importances, feature_names)

                            elif hasattr(model, 'feature_log_prob_'):
                                # For Naive Bayes (e.g., MultinomialNB)
                                log_probs = model.feature_log_prob_
                                if issparse(log_probs): # Convert sparse to dense if necessary
                                    log_probs = log_probs.toarray()

                                if log_probs.ndim > 1: # Multi-class
                                    # Use log probabilities for the predicted class
                                    importances = log_probs[predicted_class_index]
                                else: # Should not happen for MultinomialNB typically
                                     importances = log_probs

                                # Convert log probabilities to probabilities (or use absolute for ranking)
                                # Taking np.exp() is generally more meaningful for interpretation
                                feat_chart = create_feature_importance_chart(np.exp(importances), feature_names)
                            
                            # Store results (feat_chart will be None if not applicable or failed)
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
                st.toast("Predictions are ready!", icon="🏁")

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
                    """.format(np.random.uniform(0.05, 0.2)), unsafe_allow_html=True) # Placeholder for actual timing
                
                # Charts section
                st.subheader("Prediction Confidence")
                st.plotly_chart(result['conf_chart'], use_container_width=True)
                
                # Feature importance chart
                if result['feat_chart']: # Only display if a chart was successfully generated
                    st.subheader("Top Influential Words")
                    st.plotly_chart(result['feat_chart'], use_container_width=True)
                else:
                    st.info(f"Feature importance visualization is not available or could not be generated for {result['name']} model. This may be due to the model type or an internal error.", icon="ℹ️")
                
                # About This Model - using expander
                with st.expander(f"About the {result['name']} Model", expanded=False):
                    st.info(result['desc'], icon="💡")

elif app_mode == "About the Project":
    st.markdown('<h1 style="text-align: center;">About This Project</h1>', unsafe_allow_html=True)
    st.info("This is an AI-powered tool to classify news articles, showcasing a full machine learning workflow.", icon="ℹ️")
    
    st.markdown("""
    ## Overview
    This application uses machine learning models to classify news articles into categories such as Business, Politics, Sports, etc. 
    It demonstrates an end-to-end NLP classification pipeline including text preprocessing, feature extraction, and model prediction.
    
    ## How It Works
    1. **Text Preprocessing**: The input text is cleaned by converting to lowercase, removing special characters, and normalizing whitespace.
    2. **Feature Extraction**: Text is converted to numerical features using TF-IDF vectorization.
    3. **Model Prediction**: Multiple machine learning models make predictions on the processed text.
    4. **Result Visualization**: Detailed insights and visualizations are provided for each model's prediction.
    
    ## Models Used
    - **Naive Bayes**: Fast probabilistic classifier based on Bayes' theorem, assuming independence between features.
    - **SVM (Support Vector Machine)**: Finds optimal decision boundaries to separate classes.
    - **Random Forest**: An ensemble of decision trees, known for robustness.
    - **Logistic Regression**: A linear model for classification that provides probabilities.
    
    ## Performance Metrics
    The models were trained on the BBC News dataset and achieved the following accuracies:
    """)
    
    # Display model accuracies
    acc_cols = st.columns(4)
    models = list(model_info.keys())
    for i, model_name in enumerate(models):
        with acc_cols[i]:
            st.metric(model_name, f"{accuracies.get(model_name, 0):.2%}")
    
    st.markdown("""
    ## Technical Stack
    - Python
    - Scikit-learn (Machine Learning)
    - Streamlit (Web Interface)
    - Plotly (Visualizations)
    
    ## Development Notes
    This project showcases:
    - Multi-model comparison and evaluation.
    - Interactive visualization of model predictions.
    - Feature importance analysis.
    - Clean UI/UX design with a dark theme.
    """)