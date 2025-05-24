"""
Fixed Grad-CAM implementation for transfer learning models
"""
import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, img_path=None):
    """
    Generate Grad-CAM visualization for model explainability
    
    Args:
        model: Trained model
        img_array: Preprocessed image array
        img_path: Original image path (optional)
        
    Returns:
        Original image and heatmap overlay
    """
    # Get the last convolutional layer
    last_conv_layer = None
    
    # For transfer learning models with MobileNetV2
    if isinstance(model.layers[0], tf.keras.Model):
        base_model = model.layers[0]
        # Find the last convolutional layer in the base model
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
    else:
        # For custom models
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
    
    if last_conv_layer is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create a model that maps the input image to the activations
    # of the last conv layer and output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
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
    if img_path:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.uint8(img_array[0] * 255)
        
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return img, superimposed_img
