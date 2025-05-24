"""
Training script for the Fruit Quality Classifier model
"""
import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import FruitQualityClassifier

def create_sample_data(data_dir, num_samples=20):
    """
    Create sample fruit images for testing the training pipeline
    
    Args:
        data_dir: Directory to save sample images
        num_samples: Number of samples per class
    """
    # Create directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'train/good'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train/average'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train/bad'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val/good'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val/average'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val/bad'), exist_ok=True)
    
    # Generate sample images for each class
    for class_name in ['good', 'average', 'bad']:
        # Different colors for different classes
        if class_name == 'good':
            color = (0, 255, 0)  # Green for good
        elif class_name == 'average':
            color = (255, 255, 0)  # Yellow for average
        else:
            color = (0, 0, 255)  # Red for bad
            
        # Create training samples
        for i in range(num_samples):
            # Create a blank image
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Draw a circle (fruit)
            cv2.circle(img, (50, 50), 40, color, -1)
            
            # Add some noise/spots for bad and average fruits
            if class_name == 'bad':
                for _ in range(10):
                    x = np.random.randint(30, 70)
                    y = np.random.randint(30, 70)
                    cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
            elif class_name == 'average':
                for _ in range(3):
                    x = np.random.randint(30, 70)
                    y = np.random.randint(30, 70)
                    cv2.circle(img, (x, y), 3, (0, 0, 0), -1)
            
            # Save the image
            cv2.imwrite(os.path.join(data_dir, f'train/{class_name}/sample_{i}.jpg'), 
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Create validation samples (slightly different)
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.circle(img, (50, 50), 35, color, -1)
            
            if class_name == 'bad':
                for _ in range(8):
                    x = np.random.randint(30, 70)
                    y = np.random.randint(30, 70)
                    cv2.circle(img, (x, y), 4, (0, 0, 0), -1)
            elif class_name == 'average':
                for _ in range(2):
                    x = np.random.randint(30, 70)
                    y = np.random.randint(30, 70)
                    cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
            
            # Save the validation image
            if i < num_samples // 2:  # Use half for validation
                cv2.imwrite(os.path.join(data_dir, f'val/{class_name}/sample_{i}.jpg'), 
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def train_model(args):
    """
    Train the Fruit Quality Classifier model
    
    Args:
        args: Command line arguments
    """
    # Create sample data if needed
    if args.create_samples:
        print("Creating sample data...")
        create_sample_data(args.data_dir, args.num_samples)
    
    # Initialize the classifier
    classifier = FruitQualityClassifier(img_size=args.img_size, num_classes=3)
    
    # Build the model
    model = classifier.build_model(use_transfer_learning=args.transfer_learning)
    model.summary()
    
    # Create data generators
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    train_generator, validation_generator = classifier.create_data_generators(
        train_dir, val_dir, batch_size=args.batch_size
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = classifier.train(
        train_generator,
        validation_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save the final model
    classifier.save_model(os.path.join(args.model_dir, 'fruit_quality_model.h5'))
    
    # Plot training history
    classifier.plot_training_history(history)
    plt.savefig(os.path.join(args.model_dir, 'training_history.png'))
    
    # Test prediction on a sample image
    sample_image_path = os.path.join(args.data_dir, 'train/good/sample_0.jpg')
    if os.path.exists(sample_image_path):
        result = classifier.predict(sample_image_path)
        print(f"Sample prediction: {result}")
        
        # Generate and save Grad-CAM visualization
        img, heatmap = classifier.generate_gradcam(sample_image_path)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.savefig(os.path.join(args.model_dir, 'gradcam_visualization.png'))

def main():
    parser = argparse.ArgumentParser(description='Train Fruit Quality Classifier')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the model')
    parser.add_argument('--img_size', type=int, default=100,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='Use transfer learning with MobileNetV2')
    parser.add_argument('--create_samples', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of sample images per class')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    train_model(args)

if __name__ == '__main__':
    main()
