
from datasets import load_dataset

ds = load_dataset('beans')

#ViTFeatureExtractor (전처리 함수.)

from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

# 전처리 함수를 이용한 ds 전처리
ds = load_dataset('beans')

def transform(example_batch):
    
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

   
    inputs['labels'] = example_batch['labels']
    return inputs

prepared_ds = ds.with_transform(transform)



