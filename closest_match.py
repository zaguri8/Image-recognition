from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import cv2

# Function to extract features using VGG16
model = VGG16(weights='./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Load and preprocess database images
db_features = []
db_image_paths = []

for image_path in glob.glob(f'./images/*.png'):
   db_features.append(extract_features(image_path))
   db_image_paths.append(image_path)

db_features = np.array(db_features)

# Extract features from the query image
query_features = extract_features(f'./output/5.png')

# Calculate similarity
similarities = cosine_similarity(db_features, [query_features])

# Find the closest match
closest_image_idx = np.argmax(similarities)
closest_image_path = db_image_paths[closest_image_idx]
print(f"The closest match is {closest_image_path}")

