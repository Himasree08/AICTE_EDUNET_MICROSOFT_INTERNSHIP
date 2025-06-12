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

1. Train the Emotion Model
  python emotion_model.py
2. Predict Emotion on a New Image
Update the IMAGE_PATH in predict_image.py and run:
  python predict_image.py
3. Run VQA
Provide an image and type your question when prompted:
  python vqa_with_fallback.py

📊 Sample Output

Emotion Prediction:
  Input: PublicTest_87012441.jpg
  Output: Predicted Emotion: Happy
  
VQA:
  Input Image: PublicTest_96278259.jpg
  Question: What is the person doing?
  Answer: Smiling
  
Fallback:
  Question: What is the capital of France?
  Response: Fallback: This is out of my knowledge.

🚀 Future Improvements :
Add real-time webcam emotion detection
Deploy web interface for integrated VQA + FER
Use advanced CNNs like ResNet or EfficientNet
Expand question answering beyond simple visual facts


📚 References
FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
HuggingFace BLIP Model: https://huggingface.co/Salesforce/blip-vqa-base
PyTorch: https://pytorch.org/


