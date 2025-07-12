import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense, Add, Softmax
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# Load SetB (Training Data 2000)
def load_train_data():
    df_nsr = pd.read_csv('SetB_NSR_Normalized.csv')
    df_mi = pd.read_csv('SetB_MI_Normalized.csv')
    df_hcm = pd.read_csv('SetB_HCM_Normalized.csv')
    df_dcm = pd.read_csv('SetB_DCM_Normalized.csv')
    df_bbb = pd.read_csv('SetB_BBB_Normalized.csv')
    df_vhd = pd.read_csv('SetB_VHD_Normalized.csv')
    df_pal = pd.read_csv('SetB_PAL_Normalized.csv')
    df_drm = pd.read_csv('SetB_DRM_Normalized.csv')

    df_nsr['label'] = 0
    df_mi['label']  = 1
    df_hcm['label'] = 2
    df_dcm['label'] = 3
    df_bbb['label'] = 4
    df_vhd['label'] = 5
    df_pal['label'] = 6
    df_drm['label'] = 7

    return pd.concat([df_nsr, df_mi, df_hcm, df_dcm, df_bbb, df_vhd, df_pal, df_drm], ignore_index=True)

# Load SetA (Validation Data 1000)
def load_validation_data():
    df_nsr = pd.read_csv('SetA_NSR_Normalized.csv')
    df_mi = pd.read_csv('SetA_MI_Normalized.csv')
    df_hcm = pd.read_csv('SetA_HCM_Normalized.csv')
    df_dcm = pd.read_csv('SetA_DCM_Normalized.csv')
    df_bbb = pd.read_csv('SetA_BBB_Normalized.csv')
    df_vhd = pd.read_csv('SetA_VHD_Normalized.csv')
    df_pal = pd.read_csv('SetA_PAL_Normalized.csv')
    df_drm = pd.read_csv('SetA_DRM_Normalized.csv')

    df_nsr['label'] = 0
    df_mi['label']  = 1
    df_hcm['label'] = 2
    df_dcm['label'] = 3
    df_bbb['label'] = 4
    df_vhd['label'] = 5
    df_pal['label'] = 6
    df_drm['label'] = 7

    return pd.concat([df_nsr, df_mi, df_hcm, df_dcm, df_bbb, df_vhd, df_pal, df_drm], ignore_index=True)

# Preprocess data
def preprocess_data_separate(train_df, val_df):
    x_train = train_df.drop('label', axis=1).values
    y_train = tf.keras.utils.to_categorical(train_df['label'].values, num_classes=8)

    x_val = val_df.drop('label', axis=1).values
    y_val = tf.keras.utils.to_categorical(val_df['label'].values, num_classes=8)

    return x_train, y_train, x_val, y_val

# Residual block
def residual_block(x, filters, kernel_size=3, increase_filters=False):
    shortcut = x
    if increase_filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, shortcut])
    return x

# Model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = residual_block(inputs, 32)
    x = residual_block(x, 64, increase_filters=True)
    x = residual_block(x, 64)
    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes)(x)
    outputs = Softmax()(x)
    return Model(inputs=inputs, outputs=outputs)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'bo-', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'bo-', label='Train Loss')
    plt.plot(history.history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    train_df = load_train_data()
    val_df = load_validation_data()
    x_train, y_train, x_val, y_val = preprocess_data_separate(train_df, val_df)

    # Reshape for Conv1D
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    model = build_model((x_train.shape[1], 1), 8)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

    plot_history(history)

    # Evaluate
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, ['NSR', 'MI', 'HCM', 'DCM', 'BBB', 'VHD', 'PAL', 'DRM'])

    print(classification_report(y_true, y_pred_classes))

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f'Final Validation Accuracy: {val_accuracy * 100:.2f}%')

    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    print(f'Final Training Accuracy: {train_accuracy * 100:.2f}%')

    model.save('cnn_model.h5')
