import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Modeli yükleyelim
model = Custom(weights='custom.h5')

# Grad-CAM hesaplaması için yardımcı fonksiyonlar
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

base_dir = 'normal'
output_base_dir = 'normal2'


if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for sub_dir in sub_dirs:
    img_files = [f for f in os.listdir(sub_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    for img_file in img_files:
        img_path = os.path.join(sub_dir, img_file)
        img_array = get_img_array(img_path, size=(224, 224))

        last_conv_layer_name = "block5_conv3"

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        relative_path = os.path.relpath(sub_dir, base_dir)
        output_sub_dir = os.path.join(output_base_dir, relative_path)
        
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        cam_path = os.path.join(output_sub_dir, "cam_" + img_file)
        save_and_display_gradcam(img_path, heatmap, cam_path)
