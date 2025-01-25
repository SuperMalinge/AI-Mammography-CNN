import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError

class MammographyCNN(tf.keras.Model):
    def __init__(self, layers):
        super(MammographyCNN, self).__init__()
        self.layer_list = layers
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # Adding contrast enhancement layer for mammograms
        self.contrast_layer = tf.keras.layers.Lambda(
            lambda x: tf.image.per_image_standardization(x))
        
    def call(self, inputs):
        x = self.contrast_layer(inputs)
        for layer in self.layer_list:
            x = layer(x)
            x = self.batch_norm(x)
        return x

def preprocess_mammogram(image):
    # Specialized preprocessing for mammography
    image = tf.image.adjust_contrast(image, 2.0)  # Enhance contrast
    image = tf.image.adjust_gamma(image, 1.2)     # Gamma correction
    return normalize_mammogram(image)

def normalize_mammogram(image):
    # Specific normalization for mammography intensity values
    p2, p98 = np.percentile(image, (2, 98))
    image = (image - p2) / (p98 - p2)
    return np.clip(image, 0, 1)

# Modified layer configurations for mammography features
layer_configs = [
    (64, 5),   # Larger kernels for initial feature detection
    (64, 5),   # Mammographic mass detection
    (128, 3),  # Microcalcification detection
    (128, 3),  # Fine detail processing
    (256, 3),  # Complex tissue patterns
    (128, 3),  # Feature synthesis
    (64, 3),   # Refinement
    (32, 3),   # Detail preservation
    (1, 3)     # Final segmentation
]

# Initialize with high resolution for mammogram details
train_images = np.zeros(shape=(128, 2048, 2048, 1))  # High-res mammograms
train_labels = np.zeros(shape=(128, 2048, 2048, 1))

# Training parameters optimized for mammography
num_epochs = 200
init_lr = 0.00001  # Very low learning rate for stability
batch_size = 1     # Single image due to size

def visualize_mammogram_results(original, ground_truth, prediction, iter):
    plt.figure(figsize=(20, 6))
    
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Mammogram')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(ground_truth, cmap='jet', alpha=0.7)
    plt.title('Ground Truth Annotations')
    plt.axis('off')
    
    plt.subplot(133)
    overlay = plt.imshow(original, cmap='gray')
    plt.imshow(prediction, cmap='hot', alpha=0.5)
    plt.title('Detected Abnormalities')
    plt.axis('off')
    
    plt.savefig(f'mammogram_results/iteration_{iter}.png')
    plt.close()

# Training loop with specialized mammography processing
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size]
        
        # Apply mammography-specific preprocessing
        current_batch = preprocess_mammogram(current_batch)
        
        with tf.GradientTape() as tape:
            predictions = model(current_batch)
            loss_value = tf.reduce_mean(tf.square(predictions - current_label))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if iter % 5 == 0:
            visualize_mammogram_results(current_batch[0], 
                                     current_label[0], 
                                     predictions[0], 
                                     iter)

# Save the trained model
model.save('mammography_model')
