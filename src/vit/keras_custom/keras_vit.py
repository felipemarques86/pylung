import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vit.keras_custom.patches import Patches, PatchEncoder





def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
    bbox_layer_density=4,
    activation=None,
    dropout1=.1,
    dropout2=.1,
    dropout3=.3
):
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size, input_shape, num_patches, projection_dim, num_heads,
                  transformer_units, transformer_layers, mlp_head_units)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, patch_size, input_shape, projection_dim, num_heads,
                  transformer_units, transformer_layers, mlp_head_units)(patches)


    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout2)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout3)

    output = layers.Dense(bbox_layer_density, activation=activation)(
        features
    )

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=output)
