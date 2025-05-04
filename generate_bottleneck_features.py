import numpy as np
import os
from keras.preprocessing import image
from tqdm import tqdm
from extract_bottleneck_features import extract_Resnet50

# Example directory structure (adjust as needed)
data_dir = 'images'  # directory with all dog images
bottleneck_path = 'bottleneck_features'
os.makedirs(bottleneck_path, exist_ok=True)

# Load image tensors
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# Load all images and extract features
features = []
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

for path in tqdm(image_paths):
    tensor = path_to_tensor(path)
    feature = extract_Resnet50(tensor)
    features.append(feature)

# Stack and save
features = np.vstack(features)
np.savez(os.path.join(bottleneck_path, 'DogResnet50Data.npz'), features=features)
