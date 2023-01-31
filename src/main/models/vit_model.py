import tensorflow as tf
from tensorflow.keras import layers

from main.models.ml_model import MlModel


class VitModel(MlModel):
    def __init__(self, name, version, patch_size=0, projection_dim=0, num_heads=0, mlp_head_units=0, dropout1=0, dropout2=0,
                 dropout3=0, activation=0, transformer_layers=0, image_size=0, num_classes=0, image_channels=0):
        super().__init__(name, version, num_classes, image_size)
        self.transformer_layers = transformer_layers
        self.activation = activation
        self.dropout3 = dropout3
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.mlp_head_units = mlp_head_units
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.type = 'VitModel'
        self.image_channels = image_channels

    def export_as_table(self):
        table = super().export_as_table()
        table.add_row(["Activation", self.activation])
        table.add_row(["Transformer Layers", self.transformer_layers])
        table.add_row(["Dropout 1", self.dropout1])
        table.add_row(["Dropout 2", self.dropout1])
        table.add_row(["Dropout 3", self.dropout1])
        table.add_row(["Image Size", self.image_size])
        table.add_row(["Image Channels", self.image_channels])
        table.add_row(["Num heads", self.num_heads])
        table.add_row(["Projection Dim", self.projection_dim])
        table.add_row(["Patch Size", self.patch_size])
        table.add_row(["MLP Head Units", self.mlp_head_units])
        return table


    def process_loaded_model(self, model):
        self.transformer_layers = model.transformer_layers
        self.activation = model.activation
        self.dropout3 = model.dropout3
        self.dropout2 = model.dropout2
        self.dropout1 = model.dropout1
        self.mlp_head_units = model.mlp_head_units
        self.num_heads = model.num_heads
        self.projection_dim = model.projection_dim
        self.patch_size = model.patch_size

    def build_model(self):


        transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        input_shape = (self.image_size, self.image_size, self.image_channels)
        num_patches = (self.image_size // self.patch_size) ** 2
        inputs = layers.Input(shape=input_shape)

        # Create patches
        patches = Patches(self.patch_size, input_shape, num_patches, self.projection_dim, self.num_heads,
                          transformer_units, self.transformer_layers, self.mlp_head_units)(inputs)
        # Encode patches
        encoded_patches = PatchEncoder(num_patches, self.patch_size, input_shape, self.projection_dim, self.num_heads,
                                       transformer_units, self.transformer_layers, self.mlp_head_units)(patches)

        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=self.dropout1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=self.dropout2)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(self.dropout3)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=self.dropout3)

        output = layers.Dense(self.num_classes, activation=self.activation)(
            features
        )

        # return Keras model.
        self.model = tf.keras.Model(inputs=inputs, outputs=output)


class Patches(layers.Layer):
    def __init__(self, patch_size, p_input_shape, p_num_patches, p_projection_dim, p_num_heads, p_transformer_units,
                 p_transformer_layers, p_mlp_head_units):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.p_input_shape = p_input_shape
        self.p_num_patches = p_num_patches
        self.p_projection_dim = p_projection_dim
        self.p_num_heads = p_num_heads
        self.p_transformer_units = p_transformer_units
        self.p_transformer_layers = p_transformer_layers
        self.p_mlp_head_units = p_mlp_head_units

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": self.p_input_shape,
                "patch_size": self.patch_size,
                "num_patches": self.p_num_patches,
                "projection_dim": self.p_projection_dim,
                "num_heads": self.p_num_heads,
                "transformer_units": self.p_transformer_units,
                "transformer_layers": self.p_transformer_layers,
                "mlp_head_units": self.p_mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])


"""## Implement the patch encoding layer

The `PatchEncoder` layer linearly transforms a patch by projecting it into a
vector of size `projection_dim`. It also adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, p_patch_size, p_input_shape, p_projection_dim, p_num_heads, p_transformer_units,
                 p_transformer_layers, p_mlp_head_units):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=p_projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=p_projection_dim
        )
        self.p_input_shape = p_input_shape
        self.p_projection_dim = p_projection_dim
        self.p_num_heads = p_num_heads
        self.p_transformer_units = p_transformer_units
        self.p_transformer_layers = p_transformer_layers
        self.p_mlp_head_units = p_mlp_head_units
        self.p_patch_size = p_patch_size

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": self.p_input_shape,
                "patch_size": self.p_patch_size,
                "num_patches": self.num_patches,
                "projection_dim": self.p_projection_dim,
                "num_heads": self.p_num_heads,
                "transformer_units": self.p_transformer_units,
                "transformer_layers": self.p_transformer_layers,
                "mlp_head_units": self.p_mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
