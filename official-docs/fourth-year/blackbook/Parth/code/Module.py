#!/usr/bin/env python
# coding: utf-8

# we are testing 3 different algorithms 

# In[1]:


pip install opencv-python numpy dlib face_recognition deepface facenet-pytorch scikit-learn mtcnn


# In[2]:


pip install opencv-python

# In[3]:


pip install dlib

# In[4]:


pip install cmake

# In[5]:


!pip install dlib --no-cache-dir

# In[6]:


pip install --upgrade pip setuptools wheel

# In[7]:


pip install face_recognition

# In[8]:


!pip install deepface

# In[9]:


%pip install opencv-python scikit-learn matplotlib seaborn tensorflow

# In[10]:


import cv2
print(cv2.__version__)

# In[18]:


pip uninstall opencv-python --yes

# In[1]:


pip install opencv-contrib-python jupyter matplotlib scikit-learn numpy

# 

# In[8]:


import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
base_path = "comprehensive_db\\comprehensive_db"

# === 1. Load Data ===
def load_faces_from_folder(base_path, split='train', image_size=(100, 100)):
    faces = []
    labels = []
    person_names = []
    for person_name in os.listdir(base_path):
        person_folder = os.path.join(base_path, person_name, split)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, image_size)
            faces.append(img_resized)
            labels.append(person_name)  # use actual name for now
            person_names.append(person_name)

    return np.array(faces), np.array(labels)

# === 2. Encode Labels ===
def encode_labels(labels):
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    return numeric_labels, le


# In[14]:


X_train, y_train_names = load_faces_from_folder(base_path, 'train')
X_test, y_test_names= load_faces_from_folder(base_path, 'test')

# In[15]:


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, label_encoder):
    model.train(X_train, y_train)
    predictions = []

    for face in X_test:
        label_pred, _ = model.predict(face)
        predictions.append(label_pred)

    # Decode numeric labels back to names
    y_test_names = label_encoder.inverse_transform(y_test)
    y_pred_names = label_encoder.inverse_transform(predictions)

    print(f"--- {model_name} ---")
    print(classification_report(y_test_names, y_pred_names))
    return classification_report(y_test_names, y_pred_names, output_dict=True)


# In[16]:


y_train, label_encoder = encode_labels(y_train_names)
y_test = label_encoder.transform(y_test_names)

# Convert to correct format for OpenCV
X_train = [img for img in X_train]
X_test = [img for img in X_test]

# Create models
lbph_model = cv2.face.LBPHFaceRecognizer_create()
eigen_model = cv2.face.EigenFaceRecognizer_create()
fisher_model = cv2.face.FisherFaceRecognizer_create()

# Train and evaluate
lbph_results = train_and_evaluate_model(lbph_model, "LBPH Face Recognizer", X_train, y_train, X_test, y_test, label_encoder)
eigen_results = train_and_evaluate_model(eigen_model, "EigenFace Recognizer", X_train, y_train, X_test, y_test, label_encoder)
fisher_results = train_and_evaluate_model(fisher_model, "FisherFace Recognizer", X_train, y_train, X_test, y_test, label_encoder)

# In[17]:


print(f"Unique classes in y_train: {np.unique(y_train)}")
print(f"Number of classes: {len(np.unique(y_train))}")

# In[21]:


import os
import cv2
import numpy as np
from sklearn.metrics import classification_report

base_path = "comprehensive_db\\comprehensive_db"
image_size = (100, 100)

def load_images(folder, label, max_images=None):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if max_images and i >= max_images:
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

def train_individual_models(base_path):
    person_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for person in person_dirs:
        print(f"\nTraining model for: {person}")
        
        # Positive samples (label = 1)
        pos_train_dir = os.path.join(base_path, person, 'train')
        pos_test_dir = os.path.join(base_path, person, 'test')
        pos_train_imgs, pos_train_labels = load_images(pos_train_dir, label=1)
        pos_test_imgs, pos_test_labels = load_images(pos_test_dir, label=1)

        # Negative samples (label = 0) from other people's train dirs
        neg_train_imgs = []
        neg_train_labels = []
        neg_test_imgs = []
        neg_test_labels = []
        for other_person in person_dirs:
            if other_person == person:
                continue
            other_train_dir = os.path.join(base_path, other_person, 'train')
            other_test_dir = os.path.join(base_path, other_person, 'test')
            imgs, labels = load_images(other_train_dir, label=0, max_images=len(pos_train_imgs)//(len(person_dirs)-1))
            neg_train_imgs += imgs
            neg_train_labels += labels
            imgs, labels = load_images(other_test_dir, label=0, max_images=len(pos_test_imgs)//(len(person_dirs)-1))
            neg_test_imgs += imgs
            neg_test_labels += labels

        # Combine positives and negatives
        X_train = pos_train_imgs + neg_train_imgs
        y_train = pos_train_labels + neg_train_labels
        X_test = pos_test_imgs + neg_test_imgs
        y_test = pos_test_labels + neg_test_labels

        # Train LBPH model
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(X_train, np.array(y_train))

        # Evaluate
        predictions = []
        for face in X_test:
            label_pred, _ = model.predict(face)
            predictions.append(label_pred)

        print(classification_report(y_test, predictions, target_names=["Other", person]))

train_individual_models(base_path)


# In[32]:


import matplotlib.pyplot as plt

# Accuracy data
accuracy_per_person = {
    'abhijeet': 0.95,
    'ashmi': 0.97,
    'avishkar': 0.47,
    'khare': 0.89,
    'krish': 0.78,
    'maitreyee': 0.87,
    'mayur': 0.82,
    'naman': 0.74,
    'nishad': 0.75,
    'parth': 0.69,
    'prathamesh': 0.85,
    'sahaj': 0.48,
    'satyam': 0.84,
    'saubhagya': 0.88,
    'sourab': 0.92
}

# Prepare data
names = list(accuracy_per_person.keys())
accuracies = list(accuracy_per_person.values())
indices = list(range(1, len(names)+1))  # [1, 2, ..., N]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(indices, accuracies, color='skyblue', edgecolor='black')
plt.ylim(0, 1.05)
plt.xlabel("Person Number")
plt.ylabel("Accuracy")
plt.title("Accuracy per Person (Class) - LBPH Face Recognizer")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(indices)

# Build numbered name list for legend
legend_text = "\n".join([f"{i}. {name}" for i, name in enumerate(names, start=1)])

# Add text box to the plot
plt.text(len(indices)+1.7, 0.95, legend_text, fontsize=9, va='top', ha='left',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()


# In[ ]:



