import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# import library
from pyimagesearch import config, network, utils
from torchvision import datasets, transforms
import torch.optim as optim
import torch

import matplotlib

# change the backend based on the non-gui backend available
matplotlib.use("agg")


# basic transformation
transform = transforms.Compose(
    [transforms.Pad(2), transforms.ToTensor()]
)

# Load data set
trainset = datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.BATCH_SIZE, shuffle=True
)

# load the MNIST test data and create a dataloader
testset = datasets.MNIST(
    "data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=True
)

#kiemtra thiet bi
if config.DEVICE == "cuda":
    print("[INFO] Sử dụng GPU để training.")
    print("[INFO] GPU được sử dụng: ", torch.cuda.get_device_name(0))
else:
    print("[INFO] Sử dụng CPU để training.")

#tao encoder
encoder = network.Encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM).to(config.DEVICE)

#tao decoder
decoder = network.Decoder(config.EMBEDDING_DIM, config.SHAPE_BEFORE_FLATTENING).to(config.DEVICE)


#pass to VAE
vae = network.VAE(encoder, decoder)

#optimizer
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) ,lr=config.LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=config.PATIENCE
)


# initialize the best validation loss as infinity
best_val_loss = float("inf")

# initialize lists to track losses
train_losses = []
val_losses = []

# Lưu ảnh tái tạo trước khi bắt đầu train (mô hình chưa train)
print("[INFO] Lưu ảnh tái tạo trước khi training (mô hình chưa train)...")
utils.display_random_images(
    test_loader,
    vae,
    file_recon=config.FILE_RECON_BEFORE_TRAINING,
    file_real=config.FILE_REAL_BEFORE_TRAINING,
    title_recon="Reconstructed Images (Before Training)",
    title_real="Real Images (Before Training)",
    display_real=True,
    num_images=32,
    num_images_per_row=8,
)

# start training by looping over the number of epochs
for epoch in range(config.EPOCHS):
    # set the vae model to train mode
    # and move it to CPU/GPU
    vae.train()
    vae.to(config.DEVICE)

    running_loss = 0.0
    # loop over the batches of the training dataset
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(config.DEVICE)
        optimizer.zero_grad()

        # forward pass through the VAE
        pred = vae(data)

        # compute the VAE loss
        loss = utils.vae_loss(pred, data)

        # backward pass and optimizer step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # compute average loss for the epoch
    train_loss = running_loss / len(train_loader)
    # compute validation loss for the epoch
    val_loss = utils.validate(vae, test_loader)
    
    # store losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # print training and validation loss at every 20 epochs
    if epoch % 20 == 0 or (epoch + 1) == config.EPOCHS:
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    # save best vae model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"vae": vae.state_dict()},
            config.MODEL_WEIGHTS_PATH,
        )
    # adjust learning rate based on the validation loss
    scheduler.step(val_loss)

# plot and save training and validation loss
print(f"[INFO] Plotting training/validation loss curve to {config.LOSS_PLOT}...")
utils.plot_loss(train_losses, val_losses, config.LOSS_PLOT)
