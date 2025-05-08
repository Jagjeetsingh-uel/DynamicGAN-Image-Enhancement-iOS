
'''
import torch
import coremltools as ct
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Load the PyTorch model
# Ensure this is the same Generator model definition you used during training
class ResidualBlock(nn.Module):
    """Residual Block for the Generator"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator using Residual Blocks"""
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.middle_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)  # Add ReLU here

    def forward(self, x):
        initial = self.relu(self.initial(x))  # Apply ReLU after initial conv
        residual = self.residual_blocks(initial)
        middle = self.middle_conv(residual)
        output = self.output_conv(initial + middle)
        return output # Removed final activation

# Instantiate the model
generator_model = Generator()

# Load the weights
generator_model.load_state_dict(torch.load('generator_model.pth'))  # Replace with your .pth file
generator_model.to(DEVICE)
generator_model.eval()  # Set to evaluation mode

# 2. Define a sample input
# The input shape should match the input of your generator
input_shape = (1, 3, 256, 256)  #  [batch, channel, height, width]
dummy_input = torch.randn(input_shape)

# 3. Trace the model with example input
#   * image_input: This informs Core ML that the input is an image.
#   * red_bias/green_bias/blue_bias: These are subtracted from the input image.
#   * image_scale: The input image is multiplied by this value.
#
#   By default, Core ML expects input values in the range [0, 1].  If your input
#   data is in the range [-1, 1], set image_scale to 0.5 and all biases to 0.
#
#   In this case, the values are:
#     * image_scale: 1/2
#     * red_bias: -(0.5/0.5) = -1
#     * green_bias: -(0.5/0.5) = -1
#     * blue_bias: -(0.5/0.5) = -1
traced_model = torch.jit.trace(generator_model, dummy_input)
image_input = ct.ImageType(
    name="input",
    shape=input_shape,
    scale=1/2,  # Input is in range [-1, 1]
    bias=[-1, -1, -1]
)

# 4. Convert the traced model
ml_model = ct.convert(
    traced_model,
    inputs=[image_input],
    outputs=[ct.TensorType(name="output")], # Explicitly set the name of the output
)

# 5. Save the converted model
ml_model.save('generator_model_v5.mlpackage')  # save as .mlpackage

print("Core ML model saved as generator_model_v5.mlpackage")
'''

'''import coremltools as ct
import coremltools.converters.mil.testing_reqs

# Path to your ONNX model
onnx_model_path = "generator_model.onnx"

# Convert ONNX model to Core ML format
mlmodel = ct.convert(onnx_model_path, source="onnx", minimum_deployment_target=coremltools.target.iOS18)

# Save as MLPackage
mlpackage_path = "SuperResGAN.mlpackage"
mlmodel.save(mlpackage_path)

print(f"ML Package saved at: {mlpackage_path}")'''

import onnx

import tensorflow as tf
print(tf.__version__)
import tensorflow.keras as keras
print(keras.__version__)
import onnx2tf
onnx_model_path = "generator_model.onnx"
tf_model_path = "super_res_gan_tf"

# Convert ONNX to TensorFlow SavedModel format
onnx2tf.convert(onnx_model_path, tf_model_path)
print(f"Converted ONNX to TensorFlow: {tf_model_path}")
