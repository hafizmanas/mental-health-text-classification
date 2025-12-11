# 🧠 Mental Health Text Classification  
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
