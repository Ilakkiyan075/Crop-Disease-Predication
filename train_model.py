"""
Training script for the crop disease detection model.
This script demonstrates how to train the model with your own dataset.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import CropDiseaseModel
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_dataset():
    """
    Create a sample dataset structure for demonstration.
    In practice, you would replace this with your actual dataset loading.
    """
    print("Sample dataset structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── apple_scab/          # 200+ images")
    print("│   ├── bacterial_blight/    # 200+ images")
    print("│   ├── powdery_mildew/      # 200+ images")
    print("│   ├── leaf_spot/           # 200+ images")
    print("│   ├── rust/                # 200+ images")
    print("│   └── healthy/             # 200+ images")
    print("└── validation/")
    print("    ├── apple_scab/          # 50+ images")
    print("    ├── bacterial_blight/    # 50+ images")
    print("    ├── powdery_mildew/      # 50+ images")
    print("    ├── leaf_spot/           # 50+ images")
    print("    ├── rust/                # 50+ images")
    print("    └── healthy/             # 50+ images")

def load_dataset(data_dir, img_height=224, img_width=224, batch_size=32):
    """
    Load and preprocess the dataset.
    
    Args:
        data_dir (str): Path to dataset directory
        img_height (int): Image height
        img_width (int): Image width
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Training and validation datasets
    """
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Normalize pixel values
    normalization_layer = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def train_model(data_dir, epochs=20, save_path='crop_disease_model.h5'):
    """
    Train the crop disease detection model.
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        save_path (str): Path to save the trained model
    """
    print("Starting model training...")
    
    # Load dataset
    train_ds, val_ds = load_dataset(data_dir)
    
    # Create model
    model = CropDiseaseModel()
    
    # Add data augmentation for better generalization
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])
    
    # Update model with data augmentation
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model.model = keras.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(model.class_names), activation='softmax')
    ])
    
    # Compile model
    model.model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        ),
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase
    print("Starting fine-tuning...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.model.compile(
        optimizer=keras.optimizers.Adam(1e-5/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    fine_tune_epochs = 10
    total_epochs = epochs + fine_tune_epochs
    
    history_fine = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Model training completed! Saved to {save_path}")
    
    # Plot training history
    plot_training_history(history, history_fine)
    
    return model

def plot_training_history(history, history_fine=None):
    """
    Plot training history.
    
    Args:
        history: Training history
        history_fine: Fine-tuning history (optional)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if history_fine:
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model_path, test_data_dir):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path (str): Path to the trained model
        test_data_dir (str): Path to test dataset
    """
    # Load model
    model = CropDiseaseModel(model_path)
    
    # Load test data
    test_ds = keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Normalize
    normalization_layer = keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Evaluate
    test_loss, test_accuracy = model.model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=model.class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.class_names, yticklabels=model.class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    print("Crop Disease Detection Model Training")
    print("=" * 50)
    
    # Show sample dataset structure
    create_sample_dataset()
    
    print("\nTo train the model with your own dataset:")
    print("1. Organize your images according to the structure above")
    print("2. Update the data_dir path below")
    print("3. Run this script")
    
    # Example usage (uncomment when you have a dataset)
    # data_dir = "path/to/your/dataset"
    # model = train_model(data_dir, epochs=20)
    
    print("\nFor now, the system will use a pre-configured model.")
    print("You can start using the web application immediately!")
    
    # Create a basic model for demonstration
    demo_model = CropDiseaseModel()
    print("Demo model created successfully!")
