import pandas as pd
import numpy as np
import tensorflow as tf

def load_datasets():
    paths = [
        'SetA_NSR_Normalized.csv',
        'SetA_MI_Normalized.csv',
        'SetA_HCM_Normalized.csv',
        'SetA_DCM_Normalized.csv',
        'SetA_BBB_Normalized.csv',
        'SetA_VHD_Normalized.csv',
        'SetA_PAL_Normalized.csv',
        'SetA_DRM_Normalized.csv'
    ]
    datasets = []
    for path in paths:
        try:
            data = pd.read_csv(path)
            datasets.append(data)
            print(f"Loaded {path} successfully.")
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")
    combined_dataset = pd.concat(datasets, ignore_index=True)
    print("Datasets combined successfully.")
    return combined_dataset

def representative_dataset_generator(dataset):
    for index in range(100):
        data = dataset.sample(n=1).values.astype(np.float32)
        yield [tf.cast(data.reshape(1, 12, 1), dtype=tf.float32)]  

def convert_to_tflite():
    model = tf.keras.models.load_model('cnn_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    dataset = load_datasets()
    converter.representative_dataset = lambda: representative_dataset_generator(dataset)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]

    converter.inference_input_type = tf.int8  # Fix: Ensure input is INT8
    converter.inference_output_type = tf.int8  # Fix: Ensure output is INT8

    tflite_model = converter.convert()
    return tflite_model

def save_quantized_model(tflite_model, filename='cnn_quantized.tflite'):
    try:
        with open(filename, 'wb') as f:
            f.write(tflite_model)
        print(f"Quantized model saved as {filename}")
    except Exception as e:
        print(f"Failed to save the model: {str(e)}")

def main():
    try:
        tflite_model = convert_to_tflite()
        save_quantized_model(tflite_model)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
