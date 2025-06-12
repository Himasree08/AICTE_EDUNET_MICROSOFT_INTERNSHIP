from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load image
image = Image.open("PublicTest_96278259.jpg").convert('RGB')

# Input question
question = input("Ask a question about the image: ")

# Preprocess
inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

# Fallback trigger (basic version)
irrelevant_keywords = ["capital", "name", "age", "food", "who", "when", "why", "how"]

if any(word in question.lower() for word in irrelevant_keywords):
    print("Fallback: This is out of my knowledge.")
else:
    # Proceed with prediction
    print("Answer:", answer)
