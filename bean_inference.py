
# 모델 선정과 bean data로 학습한 이후 활용
from PIL import Image

# 단일 이미지 inference 함수
def predict_image(model, image_processor, image_path):
    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    predicted_class = probs.argmax().item()
    label_name = model.config.id2label[str(predicted_class)]
    return label_name

# 사용 예시
image_path = 'test_image.jpg'
print('Predicted:', predict_image(model, image_processor, image_path))