
import gradio as gr
from PIL import Image
import torch

# 단일 이미지 inference 함수
def predict_image(model, image_processor, image):
    inputs = image_processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    predicted_class = probs.argmax().item()
    return model.config.id2label[str(predicted_class)]

# Gradio 인터페이스 정의
demo = gr.Interface(
    fn=lambda img: predict_image(model, image_processor, img),
    inputs=gr.Image(type='pil'),
    outputs='text',
    title='Bean Disease Classifier'
)
demo.launch()