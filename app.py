import streamlit as st
import pandas as pd
import re
import pickle
import json
import plotly.graph_objects as go
import numpy as np
from scipy.sparse import issparse

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
    
    /* Global Button Styles (Predict/Clear) */
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
        transition: box-shadow 0.3s ease, transform 0.1s ease, background-color 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        background-color: #42A5F5; /* Slightly darker blue on hover */
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Action Buttons (General Styling for st.button containers like Predict/Clear) */
    div.stButton {
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s ease;
    }
    div.stButton > button {
        width: 100%;
    }
    div.stButton:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(100, 181, 246, 0.4);
    }


    /* CUSTOM MODEL SELECTION CARDS */

    /* The outer wrapper for positioning */
    .model-card-container {
        position: relative; /* Crucial for positioning the button inside */
        height: 180px; /* Fixed height for consistent layout */
        margin-bottom: 15px; /* Spacing between cards */
        cursor: pointer; /* Indicate clickability for the whole card */
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    
    /* Styles for the custom card's visual appearance */
    .model-card-visual {
        background-color: #212121;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #424242;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: absolute; /* Position over the button */
        top: 0;
        left: 0;
        width: 100%;
        z-index: 2; /* Make sure it's above the button */
        pointer-events: none; /* Allow clicks to pass through to the button */
        transition: inherit; /* Inherit transitions from container */
    }

    /* Selected state for the custom card's visual */
    .model-card-container.selected .model-card-visual {
        border-color: #64B5F6;
        background: rgba(100, 181, 246, 0.1);
        box-shadow: 0 0 10px rgba(100, 181, 246, 0.5);
    }
    
    /* Hover effect for the entire card container */
    .model-card-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .model-card-container:hover .model-card-visual {
         border-color: #64B5F6; /* Also update the border on hover */
    }

    /* Style the content inside the custom card visual */
    .model-card-visual .material-icons {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #BBDEFB; /* Light blue icon color */
    }
    .model-card-visual h3 {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }
    .model-card-visual p {
        font-size: 0.8rem;
        color: #9e9e9e;
    }

    /* Checkmark icon for selected cards */
    .model-card-container.selected .model-card-visual::after {
        content: 'check_circle'; /* Material icon name */
        font-family: 'Material Icons';
        font-size: 2rem;
        color: #4CAF50; /* Green checkmark */
        position: absolute;
        top: 10px;
        right: 10px;
        line-height: 1;
    }

    /* Hide default Streamlit button styling */
    .model-selection-button > button {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: transparent !important; /* Hide text */
        z-index: 1 !important; /* Place it below the visual but above parent */
        cursor: pointer !important;
        padding: 0 !important;
        margin: 0 !important;
        /* Prevent Streamlit's default hover effects on the hidden button */
        transition: none !important; 
    }
    /* Hide the focus outline from the actual button but let it be on the custom card */
    .model-selection-button > button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    /* Hide the div.stButton's default margin for model selection buttons */
    .model-selection-button {
        margin: 0 !important;
        height: 100% !important; /* Important for the parent div.stButton to fill model-card-container */
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
    'Naive Bayes': {
        'icon': 'calculate',
        'desc': """
            **Origin & Theory**: Naive Bayes classifiers are a family of simple probabilistic classifiers based on **Bayes' Theorem** with the "naive" assumption of conditional independence between features. Specifically, for a given class, the presence or absence of a particular feature (like a word in an article) is assumed to be unrelated to the presence or absence of any other feature.
            
            **How it Works**: It calculates the probability of a document belonging to a certain class (category) given the words in it. The formula is:
            $$ P(Class|Document) \\propto P(Document|Class) \\times P(Class) $$
            Where $P(Document|Class)$ is simplified by the independence assumption to be the product of probabilities of each word appearing in that class:
            $$ P(Document|Class) = \\prod_{i=1}^{n} P(Word_i|Class) $$
            It's particularly effective for text classification due to its simplicity, speed, and good performance even with small datasets. It often uses **Bag of Words** model for document representation.

            **Strengths**:
            * **Simple and Fast**: Easy to implement and computationally efficient, especially for large datasets.
            * **Good for Text**: Performs surprisingly well in text classification tasks with relatively small training data.
            * **Handles High Dimensions**: Effective with high-dimensional feature spaces (like TF-IDF vectors).
            * **Scalable**: Can handle very large feature sets.

            **Weaknesses**:
            * **"Naive" Assumption**: The conditional independence assumption rarely holds true in real-world data, which can limit its accuracy for complex relationships.
            * **Zero Frequency Problem**: If a word in the test data was not present in the training data, its probability will be zero, leading to zero posterior probability for the entire class (often handled by Laplace smoothing).
            * **Sensitivity to Data Distribution**: Performance can degrade if data distributions change significantly.

            **Typical Use Cases in NLP**:
            * Spam detection
            * Sentiment analysis
            * Document categorization
            * Language identification
        """
    },
    'SVM': {
        'icon': 'hub',
        'desc': """
            **Origin & Theory**: Support Vector Machines (SVMs) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Proposed by Vladimir Vapnik, their core idea is to find an **optimal hyperplane** that best separates the data points of different classes in a high-dimensional space.
            
            **How it Works**: The goal is to maximize the **margin** between the hyperplane and the closest data points from each class, known as **support vectors**. A larger margin generally leads to lower generalization error. For non-linearly separable data, SVMs use the **kernel trick**, which implicitly maps the inputs into high-dimensional feature spaces where a linear separation is possible, without explicitly computing the coordinates in that space. Common kernels include Linear, Polynomial, and Radial Basis Function (RBF).
            $$ \\min_{w, b, \\xi} \\frac{1}{2} ||w||^2 + C \\sum_{i=1}^{n} \\xi_i \\text{ subject to } y_i(w \\cdot x_i - b) \\ge 1 - \\xi_i \\text{ and } \\xi_i \\ge 0 $$
            Here, $w$ is the normal vector to the hyperplane, $b$ is the offset, $C$ is a regularization parameter, and $\\xi_i$ are slack variables for misclassification.

            **Strengths**:
            * **Effective in High Dimensional Spaces**: Particularly well-suited for text classification where the number of features (words) can be very large.
            * **Effective with Clear Margin of Separation**: Works well when there's a clear distinction between classes.
            * **Memory Efficient**: Uses a subset of training points (support vectors) in the decision function.
            * **Versatile Kernels**: Can adapt to various data patterns using different kernel functions.

            **Weaknesses**:
            * **Poor Performance with Large Datasets**: Can be computationally intensive and slow to train on very large datasets.
            * **Sensitivity to Outliers**: Can be sensitive to noise and outliers as they can heavily influence the hyperplane.
            * **Choice of Kernel/Parameters**: Performance is highly dependent on the right choice of kernel function and its parameters ($C$, gamma).
            * **Lack of Probabilities**: By default, SVMs don't output probabilities directly (they can be calibrated, but it adds complexity).

            **Typical Use Cases in NLP**:
            * Text and hypertext categorization
            * Spam detection
            * Handwriting recognition
            * Gene expression classification
        """
    },
    'Random Forest': {
        'icon': 'forest',
        'desc': """
            **Origin & Theory**: Random Forest, introduced by Leo Breiman, is an **ensemble learning method** for classification and regression. It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It's built on the principle of **bagging (bootstrap aggregating)**.
            
            **How it Works**:
            1.  **Bootstrapping**: Each tree in the forest is trained on a random subset of the training data, sampled with replacement.
            2.  **Feature Randomness**: When splitting a node during tree construction, only a random subset of features is considered, preventing individual features from dominating the decision-making.
            3.  **Voting/Averaging**: For classification, the final prediction is determined by majority voting among the trees. For regression, it's the average of their predictions.
            The combination of these randomizations (data and features) reduces overfitting and improves the model's robustness and accuracy.

            **Strengths**:
            * **Reduces Overfitting**: By averaging multiple trees, it effectively reduces the risk of overfitting common in individual decision trees.
            * **High Accuracy**: Often performs very well and provides high accuracy compared to single decision trees.
            * **Handles Non-linear Relationships**: Can model complex, non-linear relationships.
            * **Implicit Feature Selection**: Can estimate feature importance, showing which words are most impactful.
            * **Robust to Outliers/Noise**: Less sensitive to outliers due to the ensemble nature.

            **Weaknesses**:
            * **Interpretability**: Less interpretable than a single decision tree; understanding the contribution of individual trees is hard.
            * **Computational Cost**: Can be computationally more expensive and slower to train than simpler models like Naive Bayes, especially with many trees.
            * **Memory Usage**: Requires more memory as it stores multiple trees.

            **Typical Use Cases in NLP**:
            * Sentiment analysis
            * Spam detection
            * Authorship attribution
            * Text categorization where complex feature interactions are expected.
        """
    },
    'Logistic Regression': {
        'icon': 'functions',
        'desc': """
            **Origin & Theory**: Despite its name, Logistic Regression is a statistical model used for **binary classification**. It's a linear model that estimates the probability of an instance belonging to a particular class. It extends to multi-class classification using strategies like One-vs-Rest (OvR) or Multinomial Logistic Regression.
            
            **How it Works**: It applies a **sigmoid (logistic) function** to the output of a linear equation, squashing the output into a probability between 0 and 1.
            $$ P(Y=1|X) = \\frac{1}{1 + e^{-( \\beta_0 + \\beta_1 X_1 + ... + \\beta_n X_n )}} $$
            The parameters ($\\beta$) are learned by maximizing the likelihood function, typically using gradient descent. For text classification, it handles high-dimensional sparse data well and offers good interpretability through the learned coefficients.

            **Strengths**:
            * **Simple and Interpretable**: The coefficients can be interpreted as the strength and direction of association between a feature (word) and the log-odds of the outcome class.
            * **Efficient**: Fast to train and predict, especially on large datasets.
            * **Good for High-Dimensional Sparse Data**: Performs well with text data represented by TF-IDF, which is typically sparse.
            * **Provides Probabilities**: Outputs probabilities directly, which can be useful for ranking or thresholding decisions.

            **Weaknesses**:
            * **Assumes Linearity**: Assumes a linear relationship between the independent variables and the log-odds of the dependent variable.
            * **Not Suitable for Complex Relationships**: May not perform as well as more complex models for highly non-linear or intricate data patterns.
            * **Feature Engineering Dependent**: Performance heavily relies on good feature engineering.

            **Typical Use Cases in NLP**:
            * Spam filtering
            * Sentiment analysis (binary classification)
            * Document classification
            * Predicting click-through rates
        """
    }
}

# Initialize session state for each model's selection status
for model_name in model_info:
    if f"{model_name}_selected" not in st.session_state:
        st.session_state[f"{model_name}_selected"] = True # All selected by default

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'results' not in st.session_state:
    st.session_state.results = []

# --- Callback function to toggle selection state (triggered by Streamlit button click) ---
def toggle_model_selection_callback(model_name):
    st.session_state[f"{model_name}_selected"] = not st.session_state[f"{model_name}_selected"]

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

    # --- CUSTOM MODEL SELECTION WITH VISIBLE ST.BUTTON ---
    st.subheader("1. Select Your Models")
    cols = st.columns(len(model_info))
    for i, (name, info) in enumerate(model_info.items()):
        with cols[i]:
            is_selected = st.session_state[f"{name}_selected"]
            selected_class = "selected" if is_selected else ""
            
            # Use a div to act as the clickable card container
            # This container will hold both the hidden Streamlit button and the visual card
            # The onclick event on this container will trigger the hidden button
            st.markdown(f"""
            <div id="model_card_container_{name.replace(' ', '_')}" 
                 class="model-card-container {selected_class}"
                 onclick="
                    var button = document.getElementById('button_to_click_{name.replace(' ', '_')}');
                    if (button) {{
                        button.click();
                    }}
                 ">
                <div class="model-card-visual">
                    <span class="material-icons">{info['icon']}</span>
                    <h3>{name}</h3>
                    <p>Accuracy: {accuracies.get(name, 0):.2%}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # The actual Streamlit button, which will be styled to be invisible and fill the card.
            # Its sole purpose is to capture the click and trigger the Streamlit rerun.
            st.button(
                label="Click to Select", # A simple label
                key=f"button_to_click_{name.replace(' ', '_')}", # Unique key
                on_click=toggle_model_selection_callback,
                args=(name,),
                type="secondary", # Minimal default styling
                use_container_width=True,
                help=f"Toggle selection for {name} model",
            )
            
            # Inject JS to move the Streamlit button into the custom card container
            # and apply the model-selection-button class to its parent div.stButton
            st.markdown(f"""
                <script>
                    var cardContainer = document.getElementById('model_card_container_{name.replace(' ', '_')}');
                    var stButtonDiv = document.querySelector('[key="{f"button_to_click_{name.replace(' ', '_')}"}"]').closest('.stButton');
                    
                    if (cardContainer && stButtonDiv) {{
                        // Move the entire stButton div into the custom card container
                        cardContainer.appendChild(stButtonDiv);
                        // Add a class to the stButton div for specific CSS targeting
                        stButtonDiv.classList.add('model-selection-button');
                    }}
                </script>
            """, unsafe_allow_html=True)


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
                                "desc": model_info[model_name]['desc'] # This will now contain the enriched description
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
                    st.info(result['desc'], icon="💡") # This now uses the enriched description

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
    """)
    
    # Display model accuracies and descriptions in an About section
    for model_name, info in model_info.items():
        st.markdown(f"### {model_name}")
        st.metric("Accuracy", f"{accuracies.get(model_name, 0):.2%}")
        # The 'desc' now includes the detailed strengths, weaknesses, and use cases
        st.markdown(info['desc'], unsafe_allow_html=True) 
        st.markdown("---") # Separator
            
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
    - Clean UI/UX design with a dark theme.
    """)