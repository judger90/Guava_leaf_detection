import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras


IMAGE_SIZE = 224
BATCH_SIZE = 10
EPOCHS = 20
NUM_CLASSES = 5
IMG_NUM = 6000
DIM = (IMAGE_SIZE, IMAGE_SIZE)

PATH = 'dataset/'
classes = ['Anthracnose', 'Canker', 'Dot', 'Healthy', 'Rust']


# Veri hazırlama
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = datagen.flow_from_directory(PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                   batch_size=BATCH_SIZE, class_mode='categorical',
                                              subset='validation', shuffle=False)

# Callbacks
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, verbose=1, min_delta=1e-4, min_lr=1e-6, mode='max')
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5, mode='max', restore_best_weights=True, verbose=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks = [reduce_lr, checkpointer]

model1_base = applications.InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights=None, include_top=False)
model1_out = layers.Flatten()(model1_base.output)
model1_out = layers.Dense(1024, activation='relu')(model1_out)
model1_out = layers.Dense(512, activation='relu')(model1_out)
model1_out = layers.Dense(3, activation='softmax', name='inception')(model1_out)
model_inception = models.Model(inputs=model1_base.input, outputs=model1_out)
model_inception.load_weights('inc.h5')

model2_base = applications.ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights=None, include_top=False)
model2_out = layers.Flatten()(model2_base.output)
model2_out = layers.Dense(1024, activation='relu')(model2_out)
model2_out = layers.Dense(512, activation='relu')(model2_out)
model2_out = layers.Dense(3, activation='softmax', name='resnet')(model2_out)
model_resnet = models.Model(inputs=model2_base.input, outputs=model2_out)
model_resnet.load_weights('resnet.h5')


inc = model_inception.predict(validation_generator)
res = model_resnet.predict(validation_generator)


# Stack the predictions
stacked_preds = np.hstack((res,inc))

# True labels
true_labels = validation_generator.classes

# Create and train stack model
stack_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(stacked_preds.shape[1],)),
    tf.keras.layers.Dense(4096, activation=tf.keras.layers.Activation(tf.nn.gelu)),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
stack_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
stack_model.fit(stacked_preds, tf.keras.utils.to_categorical(true_labels), epochs=1, batch_size=BATCH_SIZE)

# Get predictions from the stack model
stacked_preds_labels = np.argmax(stack_model.predict(stacked_preds), axis=1)

# Compute metrics
accuracy = accuracy_score(true_labels, stacked_preds_labels)
f1 = f1_score(true_labels, stacked_preds_labels, average='weighted')
precision = precision_score(true_labels, stacked_preds_labels, average='weighted')
recall = recall_score(true_labels, stacked_preds_labels, average='weighted')
auc = roc_auc_score(tf.keras.utils.to_categorical(true_labels), stack_model.predict(stacked_preds), average='weighted', multi_class='ovo')
mcc = matthews_corrcoef(true_labels, stacked_preds_labels)
bas = balanced_accuracy_score(true_labels, stacked_preds_labels)
conf_matrix = confusion_matrix(true_labels, stacked_preds_labels)

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'AUC: {auc}')
print(f'MCC: {mcc}')
print(f'Balanced Accuracy Score: {bas}')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
