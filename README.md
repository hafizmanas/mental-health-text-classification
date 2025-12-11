🧠 Mental Health Text Classification Project

A Neural NLP project focused on classifying mental-health related text into predefined psychological categories using deep learning models.
Statements from users are analyzed to predict conditions such as Anxiety, Depression, Bipolar, Stress, Suicidal Thoughts, Personality Disorder, and Normal.

📘 Project Overview
Objective

Classify raw conversational statements into mental-health categories to support early detection and awareness.

Approach

The project applies:

Text preprocessing & normalization

Word embeddings (GloVe)

Deep sequence models (BiLSTM)

Fine-tuning techniques

Evaluation metrics (Accuracy, F1-score, Precision, Recall)

A Gradio-based Web Interface for real-time predictions

🧠 Models Implemented
BiLSTM + GloVe Embeddings (Primary Model)

Uses pretrained GloVe word embeddings

Bidirectional LSTM architecture

Fine-tuned for improved accuracy

Lightweight, fast, and deployable

Saved in multiple formats: .keras, .h5, .onnx, .tflite, SavedModel

Other Models (Experimented)

Although the final model uses BiLSTM, experiments with the following were planned:

Transformer-based models

FastText embeddings

Word2Vec embeddings

(These were not used in the final deployment but are mentioned as per instructor requirements.)

✨ Features

Predict mental-health categories from raw text input

Cleaned + preprocessed dataset included

Embedding-based deep learning model

Comprehensive evaluation metrics

Gradio UI for live inference

Exported model files for cross-platform deployment

📁 Repository Contents
File / Folder	Description
*.ipynb	Notebooks for preprocessing, training, evaluation, and deployment.
bilstm_final_model.keras	Final trained BiLSTM model.
tokenizer.json	Tokenizer used during training.
bilstm_model.h5, .onnx, .tflite	Exported model formats.
savedmodel/	TensorFlow SavedModel directory.
cleaned_data.csv	Preprocessed dataset ready for model input.
utils/	Helper functions (if included).
🚀 Usage
1. Clone the Repository
git clone <repository-url>
cd <repository-folder>

2. Open the notebook

Use Kaggle, Google Colab, or Jupyter Notebook.

3. Load the model and tokenizer

The BiLSTM model (.keras format) and tokenizer are included in the repository.

4. Run Inference

Enter any text such as:

"I feel hopeless and tired lately."


The model outputs:

Predicted Class: depression
Confidence: 0.81

5. Launch the Gradio Web App

A simple Gradio interface allows live predictions:

demo.launch(share=True)


This generates a temporary public link to test the model.
