from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import time


start = time.time()

 # Adatok betöltése
PATH=r'F:/Új mappa'

 # változók hozzárendelése a képekhez   
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # könyvtár a kiképző macska képeinkkel
train_dogs_dir = os.path.join(train_dir, 'dogs')  # könyvtár a kiképző kutya képeinkkel
validation_cats_dir = os.path.join(validation_dir, 'cats')  # könyvtár az érvényesítési macska képekkel
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # könyvtár az érvényesítési kutya képekkel

 # Nézzük meg, hogy hány macska és kutya kép van az oktatási és érvényesítési könyvtárban

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val


print('Képzési macska képek:', num_cats_tr)
print('Képzési kutya képek:', num_dogs_tr)

print('összes validációs macskakép:', num_cats_val)
print('összes validációs kutyaképek:', num_dogs_val)
print("--")
print("Összes tesztelési kép:", total_train)
print("Összes validációs kép:", total_val)

 # A kényelem érdekében állítson be változókat, amelyek felhasználhatók az adatkészlet előfeldolgozása és a hálózat képzése során

batch_size = 128
epochs = 10 # 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generátor az edzési adatokhoz
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generátor a validálási adatokhoz

print("--")

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

print("--")

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

 # Képek megjelenjtése
sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
 # Modell létrehozása szekvenciálisan
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#A hálózat kiképzési rétegét
model.summary()

#A hálózat kiképzése

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

 # A hálózat kiképzésének megjelenítése
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Edzési pontosság')
plt.plot(epochs_range, val_acc, label='Validációs Pontosság')
plt.legend(loc='lower right')
plt.title('Képzési és érvényesítési pontosság')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Tanítási veszteség')
plt.plot(epochs_range, val_loss, label='Validációs veszteség')
plt.legend(loc='upper right')
plt.title('Tanítási és Validációs veszteség')
plt.show()

 # A fenti táblázatokban az edzés pontossága az idő múlásával lineárisan növekszik, míg az érvényesítési pontosság az edzési folyamat körülbelül 70% -át kiteheti.
 # Ugyanakkor észrevehető a különbség az edzés és az érvényesítési pontosság közötti pontosságban is - ez a túlteljesítés jele .

 # Ha kevés képzési példa található, a modell időnként a zajoktól vagy a képzési példák nem kívánt részleteitől tanul - olyan mértékben, hogy az negatív hatással van
 # a modell teljesítményére új példák esetén. Ezt a jelenséget túlfűtésnek hívják. Ez azt jelenti, hogy a modellnek nehéz lesz általánosítani egy új adatkészletet.

 # A túllépés elleni küzdelemnek számos módja van az edzési folyamatban. Ebben az oktatóanyagban az adatok kibővítését fogja használni, és modellünket növeli.


 # Az adatok bővítése és megjelenítése

image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

 # Használja újra a definiált és használt ugyanazt az egyedi ábrázolási funkciót
 # a fenti képfeliratokkal

plotImages(augmented_images)

 # Véletlenszerűen forgatás

image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

 # Alkalmazza a zoom nagyítást

 # zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

 # Összegzés
 
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

 # Érvényesítési adatgenerátor
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')

 # Kidobni
 # A túlfűtés csökkentésének másik módja a kimaradás bevezetése a hálózatba. Ez egy olyan szabályosítási forma, amely a hálózat súlyait csak kis értékekre kényszeríti,
 # ami szabályosabbá teszi a súlyértékek eloszlását, és a hálózat kisebb edzési példák esetén csökkenti a túlterhelést.
 # A kimaradás az ebben az oktatóprogramban alkalmazott egyik normalizálási technika

 # Ha egy rétegre vetíti a vizet, akkor az véletlenszerűen kivezeti (nullára állítja) a kimeneti egységek számát az alkalmazott rétegből az edzési folyamat során.
 # A lemorzsolódás törtszámot vesz fel bemeneti értékként, például 0,1, 0,2, 0,4 stb. Formában. Ez azt jelenti, hogy a kimeneti egységek 10%, 20% vagy 40% -a
 # véletlenszerűen esik ki az alkalmazott rétegből.

 # Ha 0,1 lemorzsolódást alkalmazunk egy adott rétegre, véletlenszerűen elpusztítja a kimeneti egységek 10% -át minden edzési korszakban.

 # Hozzon létre egy hálózati architektúrát ezzel az új kimaradási funkcióval, és alkalmazza azt különböző konvolúciókra és teljesen csatlakoztatott rétegekre.

 # Új hálózat létrehozása a kimaradókkal

 # Itt a lemorzsolást alkalmazza az első és az utolsó maximális medencerétegre. A lemorzsolódás alkalmazása véletlenszerűen az idegsejtek 20% -át nullára állítja
 # minden edzési időszakban. Ez segít elkerülni az edzési adatállomány túlzott illeszkedését.

model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_new.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_new.summary()

history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

 # Plot megjelenítése

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Edzési pontosság')
plt.plot(epochs_range, val_acc, label='Validációs Pontosság')
plt.legend(loc='lower right')
plt.title('Képzési és érvényesítési pontosság')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Tanítási veszteség')
plt.plot(epochs_range, val_loss, label='Validációs veszteség')
plt.legend(loc='upper right')
plt.title('Tanítási és Validációs veszteség')
plt.show()

tf.saved_model.save(model_new, "C:/Users/Laci_B/Documents/MSC munka/2.0/Program")


# Idő számítása
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

