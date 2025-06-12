# AICTE_EDUNET_MICROSOFT_INTERNSHIP
# 🎯 Multimodal Human Emotion Analysis and Visual Question Answering System using Deep Learning

This project is a deep learning-based multimodal system that performs **Facial Emotion Recognition (FER)** and **Visual Question Answering (VQA)**. It integrates a Convolutional Neural Network (CNN) for emotion detection with a transformer-based BLIP model for answering questions about images.

## 📌 Features

- Detects human facial emotions from grayscale images (using FER2013 dataset)
- Answers natural language questions about visual content using a pretrained VQA model
- Includes a fallback mechanism for out-of-scope questions
- Modular structure: emotion model and VQA run independently or together

## 🧠 Models Used

### 1. **Facial Emotion Recognition (FER)**
- Dataset: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Model: Custom CNN with:
  - Convolutional Layers
  - MaxPooling
  - Dropout for regularization
  - Fully connected classifier layer
- Emotions detected: `['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']`

### 2. **Visual Question Answering (VQA)**
- Model: [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base)
- Framework: Hugging Face Transformers
- Handles questions like:
  - “What is the person doing?”
  - “What color is the shirt?”
- Fallback: Rejects irrelevant questions like “What is the capital of France?”

## 🛠️ Technology Stack

- **Language:** Python
- **Libraries:** PyTorch, torchvision, transformers, PIL, matplotlib
- **Frameworks:** Hugging Face Transformers



## 📁 Project Structure


