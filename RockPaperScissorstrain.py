import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

traen_dir = 'train'
vaild_dir = 'val'
tst_dir = 'test'

traen_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=[0.8, 1.2],
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)


vaild_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
tst_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


traen_generatr = traen_datagen.flow_from_directory(
    traen_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

vaild_generatr = vaild_datagen.flow_from_directory(
    vaild_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

tst_generatr = tst_datagen.flow_from_directory(
    tst_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

bsae_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

bsae_model.trainable = True
FIN_TUNE_ATT = 40 
for layer in bsae_model.layers[:-FIN_TUNE_ATT]:
    layer.trainable = False

model = models.Sequential([
    bsae_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(traen_generatr.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

erly_stopp = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
lr_schedulr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
checkpiont = ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

histri = model.fit(
    traen_generatr,
    steps_per_epoch=traen_generatr.samples // traen_generatr.batch_size,
    epochs=100,
    validation_data=vaild_generatr,
    validation_steps=vaild_generatr.samples // vaild_generatr.batch_size,
    callbacks=[erly_stopp, lr_schedulr, checkpiont]
)

model = tf.keras.models.load_model('best_model.keras')
model.save('rock_paper_scissors_model_final.h5')

print("Training complete. Best model saved as best_model.keras")

print("Evaluating on test set:")
reslts = model.evaluate(tst_generatr)
print(f"Test Loss: {reslts[0]}, Test Accuracy: {reslts[1]}")