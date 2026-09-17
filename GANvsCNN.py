##Q1
# Importation des données de cifar10
from keras.datasets import cifar10

##Q2
# Chargement du dataset
(x_train_images, x_train_labels), (x_test_images, x_test_labels) = cifar10.load_data()

##Q3
# Vérification du format des données
print("x_train_images shape: ", x_train_images.shape)
print("x_test_images shape: ", x_test_images.shape)
print("x_train_labels shape: ", x_train_labels.shape)
print("x_test_labels shape: ", x_test_labels.shape)
# Vérification alternative du format des données, moins verbose
#assert x_train_images.shape == (50000, 32, 32, 3)
#assert x_test_images.shape == (10000, 32, 32, 3)
#assert x_train_labels.shape == (50000, 1)
#assert x_test_labels.shape == (10000, 1)

##Q4
# Affichage des 10 premiers échantillons de x_train et x_test
# pyplot pour visualiser les images
import matplotlib.pyplot as plt

for i in range(10):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train_images[i])
    plt.title(f'Label: {x_train_labels[i][0]}')
    plt.axis('off')
for i in range(10):
    plt.subplot(4, 5, i + 11)
    plt.imshow(x_test_images[i])
    plt.title(f'Label: {x_test_labels[i][0]}')
    plt.axis('off')
plt.savefig('deeplearning/figures/first_10_samples.png')
plt.close()

##Q5
# Standardisation des données
x_train_images = x_train_images.astype('float32') / 255
x_test_images = x_test_images.astype('float32') / 255

##Q6
# Construction du CNN
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(32, 32, 3)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(10, activation='softmax'))

##Q7
# Affichage du sommaire du modèle
cnn.summary()

##Q8
# Compilation et entrainement du modèle
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnn.fit(x_train_images, x_train_labels, epochs=50, batch_size=50, validation_data=(x_test_images, x_test_labels), verbose=1)

##Q9
# Matrice de confusion sur x_test
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions_cnn = cnn.predict(x_test_images)
y_pred_cnn = np.argmax(predictions_cnn, axis=1)
cm_cnn = confusion_matrix(x_test_labels, y_pred_cnn)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)
disp_cnn.plot()
plt.title("Confusion Matrix CNN")
plt.savefig('deeplearning/figures/confusion_matrix_cnn.png')
plt.close()

##Q10
# Courbe de la fonction de perte pou x_train et x_test
# on a déjà importé matplotlib.pyplot plus tôt donc on s'en re-sert
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('deeplearning/figures/loss_curve.png')
plt.close()

##Q11
# Construction du RNP
rnp = Sequential()
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu', input_shape=(32*32*3,)))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=100, kernel_initializer='he_normal', activation='elu'))
rnp.add(Dense(units=10, activation='softmax'))

##Q12
# Compilation et entrainement du modèle
from keras import optimizers

# compilation du modèle
rnp.compile(optimizer=optimizers.Nadam(learning_rate=0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# reformattage les données pour le RNP
x_train_flat = x_train_images.reshape(x_train_images.shape[0], -1)
x_test_flat = x_test_images.reshape(x_test_images.shape[0], -1)
# entrainement du modèle
rnp.fit(x_train_flat, x_train_labels, epochs=50, batch_size=50, validation_data=(x_test_flat, x_test_labels), verbose=1)

##Q13
# Matrice de confusion sur x_test
# on a déjà importe scikit-learn donc on s'en re-sert
predictions_rnp = rnp.predict(x_test_flat)
y_pred_rnp = np.argmax(predictions_rnp, axis=1)
cm_rnp = confusion_matrix(x_test_labels, y_pred_rnp)
disp_rnp = ConfusionMatrixDisplay(confusion_matrix=cm_rnp)
disp_rnp.plot()
plt.title("Confusion Matrix RNP")
plt.savefig('deeplearning/figures/confusion_matrix_rnp.png')
plt.close()
