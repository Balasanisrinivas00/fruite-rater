"""
CNN Model for Fruit Quality Classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class FruitQualityClassifier:
    def __init__(self, img_size=100, num_classes=3):
        """
        Initialize the Fruit Quality Classifier
        
        Args:
            img_size: Input image size (default: 100x100)
            num_classes: Number of quality classes (default: 3 - Good, Average, Bad)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['bad', 'average', 'good']
        
    def build_model(self, use_transfer_learning=True):
        """
        Build the CNN model architecture
        
        Args:
            use_transfer_learning: Whether to use transfer learning with MobileNetV2
        """
        if use_transfer_learning:
            # Use MobileNetV2 as base model (efficient for deployment)
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
            
            # Freeze the base model layers
            base_model.trainable = False
            
            # Build the model with the MobileNetV2 base
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Build a custom CNN from scratch
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators for training and validation
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            batch_size: Batch size for training
            
        Returns:
            train_generator, validation_generator
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=10, callbacks=None):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Build and train a model first.")
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Preprocessed image ready for model input
        """
        if isinstance(image, str):
            # Load image from file path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """
        Predict quality class for a single image
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Dictionary with predicted class and confidence scores
        """
        if self.model is None:
            raise ValueError("No model available. Build or load a model first.")
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Create result dictionary with all class confidences
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidences': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def generate_gradcam(self, image):
        """
        Generate Grad-CAM visualization for model explainability
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Original image and heatmap overlay
        """
        if self.model is None:
            raise ValueError("No model available. Build or load a model first.")
        
        # Preprocess the image
        img_array = self.preprocess_image(image)
        
        # Create a model that maps the input image to the activations
        # of the last conv layer and output predictions
        last_conv_layer = None
        
        # Find the last convolutional layer
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer.name
                break
                
        if last_conv_layer is None:
            # For transfer learning model, get the last conv layer from base model
            if isinstance(self.model.layers[0], tf.keras.Model):
                base_model = self.model.layers[0]
                for layer in reversed(base_model.layers):
                    if isinstance(layer, layers.Conv2D):
                        last_conv_layer = layer.name
                        break
        
        # Create a model that maps the input image to the activations
        # of the last conv layer and output predictions
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(last_conv_layer).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        # Extract filters and gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
            
        # Average over all channels
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        
        # Load the original image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        return img, superimposed_img
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss
        
        Args:
            history: Training history from model.fit()
        """
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
