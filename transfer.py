import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D
from tensorflow.keras import applications
from tensorflow.keras import models, layers
from vit_keras import vit


IMAGE_SIZE = 224
BATCH_SIZE = 10
EPOCHS = 20
NUM_CLASS=5
IMG_NUM = 6000
DIM = (IMAGE_SIZE, IMAGE_SIZE)

PATH_TRAIN='dataset/train'
PATH_TEST='dataset/test'
classes = ['Anthracnose', 'Canker', 'Dot', 'Healthy', 'Rust']

datagen  = ImageDataGenerator(rescale = 1./255, brightness_range=[0.8, 1.2],  zoom_range=[.99, 1.01], data_format="channels_last", fill_mode="constant", horizontal_flip=True)
train_data_gen = datagen.flow_from_directory(directory=PATH_TRAIN, target_size=DIM, batch_size=IMG_NUM, shuffle=False)
train_data, train_labels = train_data_gen.next()

train_data = train_data.reshape(-1, IMAGE_SIZE , IMAGE_SIZE , 3)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

test_data_gen = datagen.flow_from_directory(directory=PATH_TEST, target_size=DIM, batch_size=300, shuffle=False)
test_data, test_labels = test_data_gen.next()

train_data = train_data.reshape(-1, IMAGE_SIZE , IMAGE_SIZE , 3)
test_data = test_data.reshape(-1, IMAGE_SIZE , IMAGE_SIZE , 3)
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

learning_rate = 1e-4
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',factor = 0.2,patience = 2,verbose = 1,min_delta = 1e-4,min_lr = 1e-6,mode = 'max')
earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',min_delta = 1e-4,patience = 5,mode = 'max',restore_best_weights = True,verbose = 1)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = './model.hdf5',monitor = 'val_accuracy',verbose = 1,save_best_only = True,save_weights_only = True,mode = 'max')
callbacks = [reduce_lr, checkpointer]

base_model=applications.Xception(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=applications.VGG16(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=applications.ResNet50(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=applications.InceptionV3(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=applications.MobileNetV2(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), weights='imagenet',include_top=False)
#base_model=applications.DenseNet121(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=applications.EfficientNetB0(input_shape=(IMAGE_SIZE , IMAGE_SIZE , 3), weights='imagenet',include_top=False)
#base_model=vit.vit_b16(image_size = IMAGE_SIZE,activation = 'softmax',pretrained = False,include_top = False, pretrained_top = False,classes = NUM_CLASS)


outmodel = base_model.output
x = layers.Flatten()(outmodel)
x = layers.Dense(768, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
out = layers.Dense(NUM_CLASS, activation='softmax')(x)  # final layer with softmax activation

model = models.Model(inputs=base_model.input, outputs=out)

#base_model.trainable = False # for transfer learning

model.summary()

model.compile(optimizer = optimizer,loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2),metrics = ['accuracy'])

history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),callbacks= callbacks, epochs=EPOCHS)

#################################
model.save('xcep.h5')
#################################


plot_model(model, to_file='model.png', show_shapes= True)
train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
xc=range(EPOCHS)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Num of Epochs')
plt.axis([0, EPOCHS, 0.1, 1.2])
plt.grid(True)
plt.legend(['train', 'validation'], loc='lower left')
plt.savefig(f"acc.png")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Num of Epochs')
plt.legend(['train', 'validation'], loc='lower left')
plt.axis([0, EPOCHS, 0.1, 2])
plt.grid(True)
plt.savefig(f"loss.png")


#train_scores= model.evaluate(train_data, train_labels)
#val_scores = model.evaluate(val_data, val_labels)
test_scores = model.evaluate(test_data, test_labels)
#print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
#print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))


#Predicting the test data
#pred_labels = model.predict(val_data)
pred_labels = model.predict(test_data)
def roundoff(arr):
    #To round off according to the argmax of each predicted label array.
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))

#Plot the confusion matrix

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)
conf_arr = confusion_matrix(test_ls, pred_ls)
plt.figure(figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)

plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.savefig(f"confusion.png")
plt.close()

print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))


# Modelin tahminlerini alın
#pred_probs = model.predict(test_data)

# İkili sınıflandırma için sadece pozitif sınıfın olasılıklarını alın
positive_probs = pred_labels[:, 1]

# Gerçek etiketlerin ikili sınıflandırma için sadece pozitif sınıf etiketlerini alın
positive_labels = test_labels[:, 1]

# ROC eğrisini ve AUC'yi hesaplayın
fpr, tpr, thresholds = roc_curve(positive_labels, positive_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizin
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(f"roc_curve.png")

print("AUC Score:", roc_auc)


