# import the necessary packages
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# set device to 'cuda' if CUDA is available and functional, 'mps' if MPS is available,
# or 'cpu' otherwise for model training and testing
def _get_device():
    if torch.cuda.is_available():
        try:
            # test actual convolution kernel execution on CUDA to detect compatibility issues
            x = torch.randn(1, 1, 4, 4).cuda()
            conv = torch.nn.Conv2d(1, 1, 2).cuda()
            conv(x)
            return "cuda"
        except Exception:
            pass
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _get_device()



# define model hyperparameters
LR = 0.001
PATIENCE = 2
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 64
EMBEDDING_DIM = 2
EPOCHS = 100
SHAPE_BEFORE_FLATTENING = (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)

# create output directory
output_dir = "output"
os.makedirs("output", exist_ok=True)

# create the training_progress directory inside the output directory
training_progress_dir = os.path.join(output_dir, "training_progress")
os.makedirs(training_progress_dir, exist_ok=True)

# create the model_weights directory inside the output directory
# for storing variational autoencoder weights
model_weights_dir = os.path.join(output_dir, "model_weights")
os.makedirs(model_weights_dir, exist_ok=True)

# define model_weights, reconstruction & real before training images paths
MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae.pt")
FILE_RECON_BEFORE_TRAINING = os.path.join(
    output_dir, "reconstruct_before_train.png"
)
FILE_REAL_BEFORE_TRAINING = os.path.join(
    output_dir, "real_test_images_before_train.png"
)

# define reconstruction & real after training images paths
FILE_RECON_AFTER_TRAINING = os.path.join(
    output_dir, "reconstruct_after_train.png"
)
FILE_REAL_AFTER_TRAINING = os.path.join(
    output_dir, "real_test_images_after_train.png"
)

# define latent space and image grid embeddings plot paths
LATENT_SPACE_PLOT = os.path.join(output_dir, "embedding_visualize.png")
IMAGE_GRID_EMBEDDINGS_PLOT = os.path.join(
    output_dir, "image_grid_on_embeddings.png"
)

# define linearly and normally sampled latent space reconstructions plot paths
LINEARLY_SAMPLED_RECONSTRUCTIONS_PLOT = os.path.join(
    output_dir, "linearly_sampled_reconstructions.png"
)
NORMALLY_SAMPLED_RECONSTRUCTIONS_PLOT = os.path.join(
    output_dir, "normally_sampled_reconstructions.png"
)
LATENT_SPACE_HISTOGRAM_PLOT = os.path.join(
    output_dir, "latent_space_histogram.png"
)
LOSS_PLOT = os.path.join(output_dir, "loss_plot.png")
RANDOM_GENERATED_IMAGES_PLOT = os.path.join(
    output_dir, "random_generated_images.png"
)
RANDOM_GENERATED_NOISE_PLOT = os.path.join(
    output_dir, "random_generated_noise.png"
)


# define class labels dictionary
CLASS_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}