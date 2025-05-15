# 🎬 Movie Review Classification with Bidirectional RNN

This project is a binary sentiment classifier built to analyze IMDB movie reviews and predict whether a given review expresses a **positive** or **negative** sentiment. It demonstrates the practical application of NLP techniques and deep learning using a real-world dataset and concludes with a deployed web interface via Streamlit.

The model uses a Bidirectional LSTM (BRNN) architecture to capture contextual meaning from both directions in a review. Earlier experiments with standard RNN and unidirectional LSTM architectures resulted in subpar performance and were removed in the current version to streamline the codebase and showcase only the most effective solution.

The project includes clear evaluation metrics, visualization of model performance across training epochs, custom threshold tuning for optimal binary classification, and real-world testing using nuanced reviews. The final model is deployed via Streamlit and publicly accessible for demonstration purposes.

## 📁 Project Structure
```
movie-review-classification/
├── Movie_review_classificationproject.ipynb   # Core notebook for model training and evaluation
├── prediction.ipynb                           # Custom review testing using saved model and threshold
├── app.py                                     # Streamlit app for deployment
├── best_model.keras                           # Saved BRNN model
├── tokenizer.pkl                              # Saved tokenizer
├── best_threshold.pkl                         # Optimal classification threshold
├── max_length.pkl                             # Max sequence length for padding
├── requirements.txt                           # Required dependencies
```

## 🚀 Features
- Real-world IMDB movie review dataset from Kaggle
- End-to-end binary classification pipeline with preprocessing, training, evaluation, and deployment
- Optimized decision threshold using AUC-ROC analysis
- Live demo deployed using Streamlit
- Visualizations of model accuracy, loss, AUC score, and ROC curve
- Manual prediction testing on nuanced custom text inputs

## 📊 Performance Metrics
- Primary evaluation: Accuracy and AUC-ROC
- Custom threshold tuning based on ROC analysis
- Final model generalizes well and performs reliably on real-world inputs

## ⚙️ How to Run Locally
1. Clone the repository:
   ```
   git clone repo
   pip install -r requirements.txt
   streamlit run app.py
   ```
## 🌐 Live Demo
Try the deployed model: https://dl-movie-review-classifier-eyuwhjnrsghu5eavf8go5i.streamlit.app/

## 🛠 Technologies Used
  Python 
  TensorFlow / Keras
  Scikit-learn
  NumPy, Pandas
  Matplotlib
  Streamlit

## 🔍 Limitations & Future Improvements
Initially experimented with simple RNN and LSTM architectures, but performance was suboptimal. Focused final model on BiLSTM, achieving higher accuracy and better ROC-AUC. Results and plots shown are for BiLSTM.
Currently supports only binary classification; future work could expand to multi-class or multi-domain sentiment tasks.
