import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


# GAN Modeli Tanımlama (224x224 boyutunda görüntüler üretmek için)
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(14 * 14 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((14, 14, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


# Discriminator Modeli
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[224, 224, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


# Kayıp Fonksiyonları
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Optimizer'lar
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Eğitim Adımı
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([images.shape[0], 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# Görüntü Üretme Fonksiyonu
def generate_images(generator, num_images, output_dir, class_name):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        noise = np.random.normal(0, 1, (1, 100))
        generated_image = generator(noise, training=False)
        generated_image = (generated_image * 127.5 + 127.5).numpy().astype(np.uint8)
        img = array_to_img(generated_image[0])
        img = img.resize((224, 224))  # Görüntüyü 224x224 boyutuna getir
        img.save(os.path.join(output_dir, f'{class_name}_gan_{i}.jpg'))


# GAN ile Görüntü Artırma
def augment_with_gan(train_dir, classes, target_count=1200, epochs=50, batch_size=32):
    generator = build_generator()
    discriminator = build_discriminator()

    for cls in classes:
        class_dir = os.path.join(train_dir, cls)
        images = [img_to_array(load_img(os.path.join(class_dir, img), target_size=(224, 224))) for img in
                  os.listdir(class_dir)]
        images = (np.array(images) - 127.5) / 127.5  # Normalize et [-1, 1] aralığına

        dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)

        print(f"{cls} sınıfı için GAN eğitiliyor...")
        for epoch in range(epochs):
            for image_batch in dataset:
                gen_loss, disc_loss = train_step(image_batch, generator, discriminator)
            print(f"Epoch {epoch + 1}/{epochs}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

        num_images_needed = target_count - len(images)
        if num_images_needed > 0:
            print(f"{cls} sınıfı için {num_images_needed} görüntü oluşturuluyor...")
            generate_images(generator, num_images_needed, class_dir, cls)


# Ana Klasörler
dataset_dir = 'dataset'
train_dir = 'train'
test_dir = 'test'

# Sınıf İsimleri
classes = ['Anthracnose', 'Canker', 'Dot', 'Healthy', 'Rust']

# Test Klasörü Oluşturma
for cls in classes:
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)

    images = os.listdir(os.path.join(dataset_dir, cls))

    if cls == 'Healthy':
        num_images = 80
    else:
        num_images = 20

    for img in images[:num_images]:
        shutil.copy(os.path.join(dataset_dir, cls, img), os.path.join(test_dir, cls, img))

    for img in images[num_images:]:
        shutil.copy(os.path.join(dataset_dir, cls, img), os.path.join(train_dir, cls, img))

# GAN ile Görüntü Artırma
augment_with_gan(train_dir, classes)

print("İşlem tamamlandı.")