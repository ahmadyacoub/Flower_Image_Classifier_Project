import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return np.expand_dims(image.numpy(), axis=0)

def predict(model, image_path, top_k=5):
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    return predictions[top_indices], [str(i) for i in top_indices]

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict the class of a flower image.")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model = load_model(args.model_path)
    class_names = load_class_names(args.category_names) if args.category_names else None

    probs, classes = predict(model, args.image_path, args.top_k)

    print("Probabilities:", probs)
    print("Classes:", classes)

    if class_names:
        print("Flower Names:", [class_names[class_index] for class_index in classes])

if __name__ == '__main__':
    main()