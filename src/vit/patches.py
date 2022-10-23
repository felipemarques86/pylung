import tensorflow as tf
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size, p_input_shape, p_num_patches, p_projection_dim, p_num_heads, p_transformer_units, p_transformer_layers, p_mlp_head_units):
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
    def __init__(self, num_patches, p_patch_size, p_input_shape, p_projection_dim, p_num_heads, p_transformer_units, p_transformer_layers, p_mlp_head_units):
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