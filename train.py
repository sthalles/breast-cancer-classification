# Authors: Thalles, Felipe, Illiana
# Defines the training procedures

import yaml
import tensorflow as tf
from model import get_model
from processing.utils import tf_record_parser, normalizer, random_rotation, random_flip_up_down, random_flip_left_right, \
    clip_image
from datetime import datetime

# define here the appropriate config file
with open(r'./from_scratch_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    print(config)

# read the config parameters
model_name = config['model_name']

# map a moel to a specific training bag
dataset_names = {'Xception': 'train_bag_2', 'DenseNet121': 'train_bag_1', 'MobileNetV2': 'train_bag_0'}

# get the model defined in the config
model = get_model(model_name, **config['model_params'])

# get the training parameters
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
learning_rate = config['train']['learning_rate']

# create the Tensorflow training dataset
train_dataset = tf.data.TFRecordDataset(filenames=['./tfrecords/clahe/' + dataset_names[model_name] + '.tfrecords'])
train_dataset = train_dataset.map(tf_record_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(normalizer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(random_rotation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(random_flip_up_down, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(random_flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(clip_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(8192)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# create the Tensorflow validation dataset
val_dataset = tf.data.TFRecordDataset(filenames=['./tfrecords/clahe/val.tfrecords'])
val_dataset = val_dataset.map(tf_record_parser)
val_dataset = val_dataset.map(normalizer)
val_dataset = val_dataset.batch(256)
val_dataset = val_dataset.repeat(1)

# define the training metrics
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

# defines the metric to be watched during training
monitor = 'val_accuracy'
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./models/' + model_name + '.h5', monitor, save_best_only=True)

# regularize the model using Early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                           mode='max',
                                                           patience=10,
                                                           restore_best_weights=True)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=METRICS)

# model = tf.keras.models.load_model('./models/' + model_name + '.h5')
# model.load_weights('./models/' + model_name + '.h5')
best_loss = model.evaluate(val_dataset)[0]

# train the model
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          callbacks=[early_stopping_callback])

val_loss = model.evaluate(val_dataset)[0]
if val_loss < best_loss:
    model.save('./models/' + model_name + '.h5')
