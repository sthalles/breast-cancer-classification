# Authors: Thalles, Felipe, Illiana

import tensorflow as tf
import os
import tempfile

def get_model_by_name(name, input_shape):
    models = {
        'MobileNetV2': lambda: tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                                 include_top=False,
                                                                 alpha=1.0,
                                                                 pooling='avg',
                                                                 weights=None),

        'DenseNet121': lambda: tf.keras.applications.DenseNet121(input_shape=input_shape,
                                                                 weights=None,
                                                                 include_top=False,
                                                                 pooling='avg'),

        'Xception': lambda: tf.keras.applications.xception.Xception(input_shape=input_shape,
                                                               weights=None,
                                                               include_top=False,
                                                               pooling='avg')
    }
    return models.get(name, lambda: None)


def get_model(model_name, **params):
    # returns the keras model defined by 'model_name'
    input_shape = params['input_shape']
    dropout = params['dropout']
    penalty = params['l2_penalty']

    # Create the base model from the pre-trained MobileNet V2
    base_model = get_model_by_name(model_name, input_shape)()

    if base_model is None:
        raise Exception("Invalid model selection. Check your config file.")

    # set all the layers to receive gradient updates
    base_model.trainable = True

    base_model = add_regularization(base_model, tf.keras.regularizers.l2(penalty), use_weights=True)

    # add Dropout and the layer classification layer
    dropout = tf.keras.layers.Dropout(dropout)
    out_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # creates and returns the model
    model = tf.keras.Sequential([
        base_model,
        dropout,
        out_layer
    ])

    return model


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001), use_weights=True):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    if use_weights:
        print("Saving...")
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights if necessary
    if use_weights:
        print("Restoring...")
        model.load_weights(tmp_weights_path, by_name=True)

    return model
