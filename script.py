import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_disease(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    prediction = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_indices[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

#image_path = 'path_to_your_image.jpg'
#model_path = 'crop_disease_model.h5'

predicted_disease, confidence = predict_disease(image_path, model_path)
print(f"Predicted disease: {predicted_disease}")
print(f"Confidence: {confidence:.2f}")