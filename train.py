import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

import shutil

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocess_training_data(dictionary):
    label = dictionary['label']
    image = dictionary['image']
    image = tf.cast(image, dtype=tf.float32) / 255.0
    image = tf.image.resize(image, [64, 64])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label

def preprocess_data(dictionary):
    label = dictionary['label']
    image = dictionary['image']
    image = tf.cast(image, dtype=tf.float32) / 255.0
    image = tf.image.resize(image, [64, 64])
    return image, label

def prepare_input_pipeline(batch_size, shuffle_buffer_size):
    train_ds = tfds.load(name='deep_weeds', split='train[:70%]', shuffle_files=True)
    valid_ds = tfds.load(name='deep_weeds',split='train[70%:85%]', shuffle_files=True)
    test_ds = tfds.load(name='deep_weeds', split='train[85%:]', shuffle_files=True)

    train_ds = train_ds.map(preprocess_training_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    valid_ds = valid_ds.shuffle(shuffle_buffer_size)
    test_ds = test_ds.shuffle(shuffle_buffer_size)
    
    train_ds = train_ds.batch(batch_size)
    valid_ds = valid_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
   
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_ds, valid_ds, test_ds

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(64, (3, 3), padding='same',  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), padding='same',  activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), padding='same',  activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same',  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(9, activation='softmax'))
    
    return model

def main():
    batch_size = 64
    num_epochs = 10000
    log_dir = './tensorboard_logs'
    
    train_ds, valid_ds, test_ds = prepare_input_pipeline(batch_size, batch_size * 3)
    
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    shutil.rmtree(log_dir) 
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './tensorboard_logs', histogram_freq = 1)
    
    model.fit(x=train_ds, epochs=num_epochs, batch_size=batch_size, validation_data=valid_ds, callbacks=[stopping_callback, tensorboard_callback])

    score = model.evaluate(test_ds)

    print('Test accuracy: ', score[1])

    model.save('model.h5')
    
if __name__ == '__main__':
    main()
