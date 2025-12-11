🧠 Mental Health Text Classification Project

A Neural NLP project focused on classifying mental-health related text into predefined psychological categories using deep learning models.
Statements from users are analyzed to predict conditions such as Anxiety, Depression, Bipolar, Stress, Suicidal Thoughts, Personality Disorder, and Normal.# 🧠 Mental Health Text Classification  
A complete NLP pipeline for classifying mental-health related text into categories such as:  
**Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal Thoughts.**

This project uses:
- **GloVe Embeddings**
- **BiLSTM Neural Network**
- **Fine-tuning for better accuracy**
- **Gradio Web App for Live Inference**

---

## 📌 **1. Project Overview**
This project aims to automatically classify conversational text into mental-health conditions.  
It processes raw text, cleans it, trains deep learning models, evaluates them, and provides a web-based UI for real-time prediction.

---

## 📂 **2. Dataset**
The dataset used in this project is **Mental Health Conversational Data**, containing text-based statements labeled with emotional or mental-health categories.

**Columns:**
- `statement` → input text  
- `status` → target class label  

After preprocessing:
- Total samples: **50,423**
- Classes:  
  - anxiety  
  - bipolar  
  - depression  
  - normal  
  - personality disorder  
  - stress  
  - suicidal  

---

## 🧹 **3. Preprocessing**
The following preprocessing steps were applied:

- Lowercasing  
- Stopword removal  
- Lemmatization  
- Tokenization  
- Padding (max length = 150)  
- Label encoding  

Final shapes:
- `X → (50423, 150)`  
- `y → (50423,)`

---

## 🧠 **4. Model Architecture**  
The primary model used:

### **BiLSTM + GloVe Embeddings**
```text
Embedding (GloVe pretrained vectors)
BiLSTM Layer (128 units × 2 directions)
Dropout
Dense (64 units, ReLU)
Dropout
Dense (Softmax Output for 7 classes)
🚀 5. Training Setup

Loss: Sparse Categorical Crossentropy
Optimizer: Adam (lr = 1e-3 → later fine-tuned to 5e-4)
Batch Size: 32
Epochs: 10
Callbacks: Early Stopping, ModelCheckpoint

The model was trained on GloVe vectors, which improves understanding of semantic meaning.

📊 6. Evaluation Results
✔ Accuracy: ~73%
✔ Macro F1-Score: ~63%

(from Cell 14 results)

✔ Precision / Recall:

Detailed in the classification report:

Class	Precision	Recall	F1
anxiety	0.78	0.69	0.73
bipolar	0.74	0.69	0.71
depression	0.71	0.68	0.69
normal	0.86	0.93	0.90
personality disorder	0.47	0.24	0.32
stress	0.44	0.48	0.46
suicidal	0.61	0.69	0.65
✔ Confusion Matrix

Shows per-class performance and misclassification patterns.

🧪 7. Testing (Inference)

A helper function was created:

label, confidence = predict_text("I feel down and stressed.")


Output:

Predicted Label: depression
Confidence: 0.81

🌐 8. Web Interface (Gradio App)

A lightweight web interface was created for demonstration.

Features:

Accepts user text input

Predicts mental-health category

Shows confidence score

Works with your trained and saved model

To launch:

demo.launch(share=True)


Kaggle generates a temporary public URL for live demo.

💾 9. Saved Model Files

The following export formats were saved:

.keras (recommended format)

.h5

.tflite

.onnx

SavedModel directory

tokenizer.json

These allow the model to be reused across frameworks and platforms.

🔬 10. Hyperparameter Experiment

A learning-rate experiment was performed:

Learning Rate	Macro F1
1e-3	0.63
1e-4	(slightly lower)

Conclusion: 1e-3 is optimal for this dataset.

📌 11. AI-Based Suggestions Applied

ChatGPT recommended:

Fine-tuning GloVe embeddings

Reducing learning rate

Increasing dropout slightly

Using balanced class weights (optional)

We applied:
✔ Fine-tuning
✔ Lower LR (5e-4)
✔ Additional dropout

This improved F1-score.

🧩 12. Folder Structure
Mental-Health-Classification/
│
├── cleaned_data.csv
├── bilstm_final_model.keras
├── tokenizer.json
├── bilstm_model.h5
├── bilstm_model.tflite
├── savedmodel/
├── notebook.ipynb
└── README.md

📘 13. How to Run

Install dependencies

Load tokenizer + model

Run the Gradio cell

Enjoy live predictions

🎉 14. Conclusion

This project successfully demonstrates a full NLP pipeline—from raw text to deep learning model to web deployment.
The BiLSTM + GloVe approach performs well for multi-class mental-health prediction.

👨‍💻 Author

Hafiz Muhammad Anas
Social Media Marketer & Software Engineering Student
Email: h8991254@gmail.com

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
