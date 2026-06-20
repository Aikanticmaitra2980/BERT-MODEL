# BERT-based Text Classification Pipeline

A streamlined PyTorch implementation for fine-tuning a pre-trained **BERT (bert-base-uncased)** model for binary text classification (e.g., sentiment analysis). 

This repository provides a modular blueprint for tokenizing text data, setting up custom PyTorch DataLoaders, fine-tuning the model using Hugging Face's `transformers` library, and evaluating model performance using `scikit-learn`.

---

## 🚀 Features

* **Custom Dataset Integration:** Efficient tokenization and padding handling using a custom PyTorch `Dataset`.
* **Stratified Data Splitting:** Complete data processing pipeline that cleanly splits inputs into Train (80%), Validation (10%), and Test (10%) sets.
* **Hugging Face Transformers:** Powered by `AutoModelForSequenceClassification` and `AutoTokenizer` for seamless model swapping.
* **Robust Evaluation:** Computes and prints Classification Accuracy, a detailed Precision/Recall/F1-score report, and a Confusion Matrix.
* **Hardware Acceleration:** Automatically detects and utilizes NVIDIA CUDA GPUs if available.

---

## 🛠️ Requirements & Installation

Make sure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
**pip install torch transformers scikit-learn pandas numpy
```output
**Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Confusion Matrix:
 [[1 0]
  [0 1]]
