import numpy as np
import tensorflow as tf
from keras import saving

#--------------------------------------------------------------------------------#
#-------------------------------- Patch creation --------------------------------#
#--------------------------------------------------------------------------------#
@saving.register_keras_serializable()
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        channels = tf.shape(images)[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(images=images, sizes=[1, self.patch_size, self.patch_size, 1], 
                                           strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        patches = tf.reshape(patches, (batch_size, num_patches_h * num_patches_w, self.patch_size**2 * channels))
        return patches

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        return (batch_size, num_patches_h * num_patches_w, self.patch_size**2 * channels)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


#--------------------------------------------------------------------------------#
#---------------------- Embedding and positional encoding -----------------------#
#--------------------------------------------------------------------------------#
@saving.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # Encodes the patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)

        # Adds positional information
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.keras.ops.expand_dims(tf.keras.ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        config.update({"projection_dim": self.projection_dim})
        return config


#--------------------------------------------------------------------------------#
#------------------------- Multilayer perceptron (MLP) --------------------------#
#--------------------------------------------------------------------------------#
def mlp(x, hidden_units):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    return x


#--------------------------------------------------------------------------------#
#------------------------ Construction of the ViViT Model -----------------------#
#--------------------------------------------------------------------------------#
def modelo_vivit(patch_size, num_patches, projection_dim, transformer_units, num_heads, transformer_layers,
                 mlp_head_units, input_shape, output_shape, dropout_rate=0.1):
    # -------------------------------- Image processing
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Create patches
    patches = tf.keras.layers.TimeDistributed(Patches(patch_size))(inputs)

    # Combine all patches into a sequence
    seq_len = input_shape[0] * num_patches
    patches = tf.keras.layers.Reshape((seq_len, patches.shape[-1]))(patches)

    # Embed patches and add positional encoding
    encoded_patches = PatchEncoder(seq_len, projection_dim)(patches)

    # Create transformer encoder layers
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Multi-Head Attention (MHA) layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads
        )(x1, x1)

        # Residual connection 1
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # Apply dropout to prevent overfitting
        x3 = tf.keras.layers.Dropout(dropout_rate)(x3)

        # Internal Multilayer Perceptron
        x3 = mlp(x3, hidden_units=transformer_units)

        # Residual connection 2
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply final layer normalization, flatten, and dropout
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(dropout_rate)(representation)

    # Final MLP
    features = mlp(representation, hidden_units=mlp_head_units)

    # Output image layer
    outputs = tf.keras.layers.Dense(int(np.prod(output_shape)), activation="sigmoid")(features)
    outputs = tf.keras.layers.Reshape(output_shape, name="output_image")(outputs)

    # Build the model by connecting inputs and outputs
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model