# NewsTagger AI: News Article Classifier

## Overview

NewsTagger AI is a Streamlit-powered web application designed to classify news articles into predefined categories using various machine learning models. This project demonstrates an end-to-end natural language processing (NLP) classification pipeline, covering text preprocessing, feature extraction, model prediction, and interactive result visualization.

The application allows users to paste news article content, select different classification models, and receive immediate category predictions along with confidence scores and insights into feature importance. It also includes a feedback mechanism to help improve model performance over time.

## Features

* **Multi-Model Classification:** Choose from several pre-trained machine learning models (Naive Bayes, SVM, Random Forest, Logistic Regression) to classify articles.
* **Real-time Prediction:** Get instant category predictions for pasted news content.
* **Confidence Scores:** Understand the certainty of each model's prediction.
* **Influential Words Analysis:** Visualize the top words that influenced a model's classification.
* **Model Comparison:** Compare predictions and confidence levels across different selected models.
* **Interactive UI:** User-friendly interface built with Streamlit and custom CSS for a modern dark theme.
* **User Feedback System:** Provide feedback on predictions to help log and potentially improve model accuracy.
* **Analytics Dashboard:** View basic analytics on collected user feedback (Correct/Incorrect classifications, most corrected categories).
* **Model Information:** Detailed descriptions and performance metrics for each integrated model.
* **Category Explanations:** Understand the scope and definition of each classification category.

## Getting Started

Follow these steps to set up and run the NewsTagger AI application on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the project files):**
    If you have Git:
    ```bash
    git clone https://github.com/Uneducated-cat/NewsTagger-AI
    cd NewsTagger-AI 
    ```
    If you downloaded a ZIP, extract it and navigate into the project directory.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Obtain Pre-trained Models and Vectorizers:**
    This application requires pre-trained machine learning models (`.pkl` files for Naive Bayes, SVM, Random Forest, Logistic Regression), a TF-IDF vectorizer (`vectorizer.pkl`), a label encoder (`label_encoder.pkl`), and a `model_accuracies.json` file. These files are typically generated by a separate training script. **Ensure these files (`vectorizer.pkl`, `label_encoder.pkl`, `model_accuracies.json`, `model_naive_bayes.pkl`, `model_svm.pkl`, `model_random_forest.pkl`, `model_logistic_regression.pkl`) are present in the root directory of your project alongside `app.py`.**
    *(If you don't have these, you would need a `train.py` script to generate them from a dataset. This README assumes they are available.)*

### Running the Application

Once all prerequisites are met and models are in place:

```bash
streamlit run app.py
```

This command will open the NewsTagger AI application in your default web browser.

## Usage

1.  **Navigate to the `Classifier` tab** (default).
2.  **Select Models:** Click on the cards of the models you wish to use for classification. You can select one or multiple models.
3.  **Paste Article Text:** Copy and paste the full content of a news article into the provided text area.
4.  **Click "Predict":** The application will process the text using the selected models and display the predictions.
5.  **Review Results:**
    * The "Prediction Details" tab shows detailed results for a selected model, including confidence scores, influential words, and options to provide feedback.
    * The "Compare Models" tab provides a side-by-side comparison of predictions from all selected models.
6.  **Provide Feedback:** Use the "Yes, it was correct" or "No, it was incorrect" buttons under each model's detailed prediction to provide feedback. If incorrect, you can select the correct category and provide an optional reason. This data is logged to `user_feedback.csv`.
7.  **Clear Results:** Use the "Clear Results" button to reset the input and predictions.
8.  **Adjust Confidence Threshold:** Use the slider in the sidebar to filter predictions based on a minimum confidence score.
9.  **Explore "About the Project":** Switch to the "About the Project" tab to learn more about the project, the models used, category definitions, training data insights, and basic feedback analytics.

## Models Used

The application integrates the following machine learning models for text classification:

* **Naive Bayes:** A probabilistic classifier known for its simplicity and efficiency, especially in text classification.
* **Support Vector Machine (SVM):** A powerful model that finds an optimal hyperplane to separate data points into different classes.
* **Random Forest:** An ensemble method that builds multiple decision trees and combines their outputs for robust predictions.
* **Logistic Regression:** A linear model that estimates the probability of an instance belonging to a particular class, commonly used for classification.

All models are trained using TF-IDF (Term Frequency-Inverse Document Frequency) features derived from the news article text.

## File Structure

```
.
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
├── vectorizer.pkl          # Pickled TF-IDF vectorizer (generated during training)
├── label_encoder.pkl       # Pickled Label Encoder (generated during training)
├── model_accuracies.json   # JSON file with model accuracies
├── model_naive_bayes.pkl   # Pickled Naive Bayes model (generated during training)
├── model_svm.pkl           # Pickled SVM model (generated during training)
├── model_random_forest.pkl # Pickled Random Forest model (generated during training)
├── model_logistic_regression.pkl # Pickled Logistic Regression model (generated during training)
└── user_feedback.csv       # (Generated after first feedback) Stores user feedback data
```

## License

This project is open-source and available under the MIT License. You are free to modify and distribute it.

```
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
