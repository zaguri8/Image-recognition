from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import numpy as np
import glob
import cv2

# Function to extract features using VGG16
model = VGG16(weights='./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

import sys
# Load and preprocess database images
db_features = []
db_image_paths = []
input_images_dir = sys.argv[1] if len(sys.argv) > 1 else './db_images'
image_extensions = ['png', 'jpg', 'jpeg']
for ext in image_extensions:
   for image_path in glob.glob(f'{input_images_dir}/*.{ext}'):
      db_features.append(extract_features(image_path))
      db_image_paths.append(image_path)

db_features = np.array(db_features)
input_image = sys.argv[2] if len(sys.argv) > 2 else './test_input/5.png'
# Extract features from the query image
query_features = extract_features(input_image)

# Calculate similarity
cosine_similarities = cosine_similarity(query_features.reshape(1, -1), db_features)
# Sort the similarities in descending order
sorted_indices = np.argsort(cosine_similarities[0])[::-1]

# Get top N most similar items
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
most_similar_indices = sorted_indices[:N]

# Retrieve the corresponding images
most_similar_images = [db_image_paths[i] for i in most_similar_indices]
# Output or return the similar_images
print(most_similar_images)  # or return similar_images

# open the images
import os
if len(sys.argv) > 4:
   super_user = sys.argv[4] 
   for image in most_similar_images:	
      os.system(f'echo {super_user} | sudo -S open {image}')


