import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from model import get_model
from processing.utils import tf_record_parser, normalizer
from utils import plot_confusion_matrix, ensemble_predict, get_eval_scores, tf_dataset_to_numpy

with open(r'./from_scratch_config.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

val_dataset = tf.data.TFRecordDataset(filenames=['./tfrecords/clahe/val.tfrecords'])
val_dataset = val_dataset.map(tf_record_parser)
val_dataset = val_dataset.map(normalizer)
val_dataset = val_dataset.batch(256)
val_dataset = val_dataset.repeat(1)

test_dataset = tf.data.TFRecordDataset(filenames=['./tfrecords/clahe/test.tfrecords'])
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.map(normalizer)
test_dataset = test_dataset.batch(256)
test_dataset = test_dataset.repeat(1)

model_names = ['Xception', 'DenseNet121', 'MobileNetV2']
models = []
for name in model_names:
    print("Model:", name)
    m = get_model(name, **config['model_params'])
    m.load_weights('./models/from-scratch/' + name + '.h5')
    # m.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.AUC()])
    # m.evaluate(test_dataset)
    models.append(m)

# get numpy arrays from tensorflow datasets
y_test = tf_dataset_to_numpy(test_dataset)
y_val = tf_dataset_to_numpy(val_dataset)

# perform ensemble prediction
y_probs = ensemble_predict(models, val_dataset)
y_pred = (y_probs > 0.5).astype(np.uint8).flatten()

# print the ensemble eval metrics
get_eval_scores(y_val, y_probs, y_pred)

class_names = np.array(['normal', 'tumor'])
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_val, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()

########################
print("----------Test----------")
y_probs = ensemble_predict(models, test_dataset)
y_pred = (y_probs > 0.5).astype(np.uint8).flatten()

# print the ensemble eval metrics
get_eval_scores(y_test, y_probs, y_pred)

# plot confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()
