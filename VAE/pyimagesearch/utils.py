#import các hàm cần thiết
import os
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib.patches import bbox_artist

from pyimagesearch import config

matplotlib.use("agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm

#chọn random ảnh
def extract_random_images(data_loader, num_images):
    all_images = []
    all_labels = []

    # cho toàn bộ ảnh từ data loader vào 
    for images, labels in data_loader:
        all_images.append(images)
        all_labels.append(labels)
        if (len(all_images) * data_loader.batch_size > 1000):
            break

    # gộp toàn bộ ảnh thành 1 tensor duy nhất
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # tạo random index/ cho ảnh và labels vào list
    num_to_extract = min(num_images, len(all_images))
    random_indices = np.random.choice(len(all_images), num_to_extract, replace=False)
    random_images = all_images[random_indices]
    random_labels = all_labels[random_indices]

    return random_images, random_labels


# hiển thị các ảnh được truyền vào , hoặc là save file
def display_images(images, labels, num_images_per_row, title, filename=None, show=True):
    # calculate the number of rows needed to display all the images

    num_rows = len(images) // num_images_per_row
    # create a grid of images using torchvision's make_grid function
    grid = torchvision.utils.make_grid(
        images.cpu(), nrow=num_images_per_row, padding=2, normalize=True
    )
    # convert the grid to a NumPy array and transpose it to
    # the correct dimensions
    grid_np = grid.numpy().transpose((1, 2, 0))
    # create a new figure with the appropriate size
    plt.figure(figsize=(num_images_per_row * 2, num_rows * 2))
    # show the grid of images
    plt.imshow(grid_np)
    # remove the axis ticks
    plt.axis("off")
    # set the title of the plot
    plt.title(title, fontsize=16)


    #add lables for images
    for i in range(len(images)):
        #cal the row and col of the current image
        row = i//num_images_per_row
        col = i % num_images_per_row
        #get the lable
        label_name = config.CLASS_LABELS[labels[i].item()]
        #add the name as text to the plot
        plt.text(
            col * (images.shape[3] + 2) + images.shape[3] // 2,
            (row + 1) * (images.shape[2] + 2) - 5,
            label_name,
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, lw=0),
        )
    #display if true
    if show:
        plt.show()
    else:
        #save if not true
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


#lấy random ảnh, pass qua vae và hiển thị
def display_random_images(
    data_loader,
    vae = None,
    file_recon=None,
    file_real=None,
    title_recon=None,
    title_real=None,
    display_real=True,
    num_images=32,
    num_images_per_row=8,
):
    #lay anh random
    random_images, random_labels = extract_random_images(data_loader, num_images)

    # if True encoder, decoder -> them anh qua VAE ,
    if vae is not None:
        # set the vae to evaluation mode
        vae.eval()
        # move the random images to the appropriate device
        random_images = random_images.to(config.DEVICE)
        # generate reconstruction image
        _, _, random_reconstructions = vae(random_images)
        # display the reconstructed images
        display_images(
            random_reconstructions.cpu(),
            random_labels,
            num_images_per_row,
            title_recon,
            file_recon,
            show=False,
        )
        # if specified, also display the original images
        if display_real:
            display_images(
                random_images.cpu(),
                random_labels,
                num_images_per_row,
                title_real,
                file_real,
                show=False,
            )
    # if no encoder and decoder are provided, simply display the original images
    else:
        display_images(
            random_images, random_labels, num_images_per_row, title="Real Images"
        )


#chạy validation và lấy loss theo hàm vae_loss
def validate(vae, test_loader):
    # set the encoder and decoder to evaluation mode
    vae.eval()

    # initialize the running loss to 0.0
    running_loss = 0.0

    # disable gradient calculation during validation
    with torch.no_grad():
        # iterate through the test loader
        for batch_idx, (data, _) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            # move the data to the appropriate device CPU/GPU
            data = data.to(config.DEVICE)
            # let data go through vae
            pred = vae(data)

            # calculate the loss between the decoded and original data
            loss = vae_loss(pred, data)
            # add the loss to the running loss
            running_loss += loss.item()

    # calculate the average loss over all batches
    # and return to the calling function
    return running_loss / len(test_loader)


#lấy ảnh pass qua embeding và trả lable + laten
def get_test_embeddings(test_loader, encoder):
    # switch the model to evaluation mode
    encoder.eval()

    # initialize empty lists to store the embeddings and labels
    points = []
    label_idcs = []

    # iterate through the test loader
    for i, data in enumerate(test_loader):
        # move the images and labels to the appropriate device
        img, label = [d.to(config.DEVICE) for d in data]
        # encode the test images using the encoder
        z_mean, _, _ = encoder(img)
        # convert the embeddings and labels to NumPy arrays
        # and append them to the respective lists
        points.extend(z_mean.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        # free up memory by deleting the images and labels
        del img, label

    # convert the embeddings and labels to NumPy arrays
    points = np.array(points)
    label_idcs = np.array(label_idcs)

    # return the embeddings and labels to the calling function
    return points, label_idcs


# từ output của embeding và lable, vẽ phân phối trong không gian
def plot_latent_space(test_loader, encoder, show=False):
    # get the embeddings and labels for the test images
    points, label_idcs = get_test_embeddings(test_loader, encoder)

    # create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))

    # create a scatter plot of the embeddings, colored by the labels
    scatter = ax.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=2.0,
        c=label_idcs,
        cmap="tab10",
        alpha=0.9,
        zorder=2,
    )

    # remove the top and right spines from the plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # add a colorbar to the plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel("Labels", rotation=270, labelpad=20)

    # if show is True, display the plot
    if show:
        # add a grid to the plot
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        plt.show()
    # otherwise, save the plot to a file and close the figure
    else:
        plt.savefig(config.LATENT_SPACE_PLOT, bbox_inches="tight")
        plt.close()



def get_random_test_images_embeddings(test_loader, encoder, imgs_visualize=5000):
    # get all the images and labels from the test loader
    all_images, all_labels = [], []
    for batch in test_loader:
        images_batch, labels_batch = batch
        all_images.append(images_batch)
        all_labels.append(labels_batch)

    # concatenate all the images and labels into a single tensor
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # randomly select a subset of the images and labels to visualize
    index = np.random.choice(range(len(all_images)), imgs_visualize)
    images = all_images[index]
    labels = all_labels[index]

    # get the embeddings for all the test images
    points, _ = get_test_embeddings(test_loader, encoder)

    # select the embeddings corresponding to the randomly selected images
    embeddings = points[index]

    # return the randomly selected images, their labels, and their embeddings
    return images, labels, embeddings


def plot_image_grid_on_embeddings(
    test_loader, encoder, decoder, grid_size=15, figsize=12, show=True
):
    # get a random subset of test images
    # and their corresponding embeddings and labels
    _, labels, embeddings = get_random_test_images_embeddings(test_loader, encoder)

    # create a single figure for the plot
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # define a custom color map with discrete colors for each unique label
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = cm.get_cmap("rainbow", num_classes)
    bounds = np.linspace(0, num_classes, num_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the scatter plot of the embeddings colored by label
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap=cmap,
        c=labels,
        norm=norm,
        alpha=0.8,
        s=300,
    )

    # Create the colorbar with discrete ticks corresponding to unique labels
    cb = plt.colorbar(scatter, ticks=range(num_classes), spacing="proportional", ax=ax)
    cb.set_ticklabels(unique_labels)

    # Create the grid of images to overlay on the scatter plot
    x = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), grid_size)
    y = np.linspace(embeddings[:, 1].max(), embeddings[:, 1].min(), grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.column_stack((xv.ravel(), yv.ravel()))

    # convert the numpy array to a PyTorch tensor
    # and get reconstructions from the decoder
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    reconstructions = decoder(grid_tensor.to(config.DEVICE))

    # overlay the images on the scatter plot
    for i, (grid_point, img) in enumerate(zip(grid, reconstructions)):
        img = img.squeeze().detach().cpu().numpy()
        imagebox = OffsetImage(img, cmap="Greys", zoom=0.5)
        ab = AnnotationBbox(
            imagebox, grid_point, frameon=False, pad=0.0, box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

    if show:
        plt.show()
    else:
        plt.savefig(config.IMAGE_GRID_EMBEDDINGS_PLOT, bbox_inches="tight")
        plt.close()


def plot_linearly_sampled_reconstructions(decoder, grid_size=15, figsize=8, show=True):
    # tạo lưới các điểm tuần tự tuyến tính trong khoảng [-3, 3]
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(3, -3, grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.column_stack((xv.ravel(), yv.ravel()))

    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        reconstructions = decoder(grid_tensor)

    # gộp các ảnh lại thành một lưới 2D lớn
    reconstructions = reconstructions.cpu()
    grid_img = torchvision.utils.make_grid(reconstructions, nrow=grid_size, padding=2, normalize=True)
    grid_np = grid_img.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Linearly Sampled Latent Space", fontsize=16)

    if show:
        plt.show()
    else:
        plt.savefig(config.LINEARLY_SAMPLED_RECONSTRUCTIONS_PLOT, bbox_inches="tight")
        plt.close()


def plot_normally_sampled_reconstructions(decoder, grid_size=15, figsize=8, show=True):
    import scipy.stats as stats
    # dùng hàm phân vị (ppf) của phân phối chuẩn để lấy mẫu grid sao cho xác suất đều nhau
    grid_x = stats.norm.ppf(np.linspace(0.01, 0.99, grid_size))
    grid_y = stats.norm.ppf(np.linspace(0.99, 0.01, grid_size))
    xv, yv = np.meshgrid(grid_x, grid_y)
    grid = np.column_stack((xv.ravel(), yv.ravel()))

    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        reconstructions = decoder(grid_tensor)

    reconstructions = reconstructions.cpu()
    grid_img = torchvision.utils.make_grid(reconstructions, nrow=grid_size, padding=2, normalize=True)
    grid_np = grid_img.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Normally Sampled Latent Space", fontsize=16)

    if show:
        plt.show()
    else:
        plt.savefig(config.NORMALLY_SAMPLED_RECONSTRUCTIONS_PLOT, bbox_inches="tight")
        plt.close()

#plot histogram phân phối các điểm trong data set khi được map vào latent
def plot_latent_space_histogram(test_loader, encoder, show=False):
    points, _ = get_test_embeddings(test_loader, encoder)
    num_dims = points.shape[1]
    
    fig, axes = plt.subplots(1, num_dims, figsize=(5 * num_dims, 5))
    if num_dims == 1:
        axes = [axes]
        
    import scipy.stats as stats
    for i in range(num_dims):
        # Vẽ histogram
        axes[i].hist(points[:, i], bins=50, density=True, alpha=0.6, color='blue')
        axes[i].set_title(f"Latent Dimension {i+1}", fontsize=14)
        axes[i].set_xlabel("Value", fontsize=12)
        axes[i].set_ylabel("Density", fontsize=12)
        
        # Vẽ đường phân phối chuẩn N(0,1) để so sánh (để xem KLD hoạt động thế nào)
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, 0, 1)
        axes[i].plot(x, p, 'k', linewidth=2, label="N(0,1)")
        axes[i].legend()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(config.LATENT_SPACE_HISTOGRAM_PLOT, bbox_inches="tight")
        plt.close()


#vẽ hàm loss
def plot_loss(train_losses, val_losses, filename=None, show=False):
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Use modern color palette
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=2.5, linestyle="-", marker="o", markersize=3, alpha=0.9)
    plt.plot(val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2.5, linestyle="-", marker="s", markersize=3, alpha=0.9)
    
    # Customizing axes and labels
    plt.xlabel("Epochs", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Loss", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("VAE Training & Validation Loss Over Epochs", fontsize=14, fontweight="bold", pad=15)
    
    # Grid customization
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Legend customization
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    if show:
        plt.show()
    elif filename is not None:
        parent_dir = os.path.dirname(filename)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


#random z và cho qua decoder
def random_generating_images(decoder, num_images=32, num_images_per_row=8, filename=None, noise_filename=None, show=True):
    decoder.eval()
    # Sample z from standard normal distribution
    z = torch.randn(num_images, config.EMBEDDING_DIM).to(config.DEVICE)
    
    with torch.no_grad():
        generated_images = decoder(z)
        
    # Calculate the number of rows needed
    num_rows = len(generated_images) // num_images_per_row
    
    # Create a grid of images using torchvision's make_grid function
    grid = torchvision.utils.make_grid(
        generated_images.cpu(), nrow=num_images_per_row, padding=2, normalize=True
    )
    
    # Convert the grid to a NumPy array and transpose it to the correct dimensions
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(num_images_per_row * 2, num_rows * 2))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Random Generated Images from Latent Space", fontsize=16)
    
    if show:
        plt.show()
    elif filename is not None:
        parent_dir = os.path.dirname(filename)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    # Plot the sampled latent vectors (noise coordinates)
    if noise_filename is not None or show:
        z_cpu = z.detach().cpu().numpy()
        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(z_cpu[:, 0], z_cpu[:, 1], color="#1f77b4", s=100, alpha=0.8, edgecolors="black", linewidths=1.5, zorder=3)
        
        # Label each point with its index in the grid
        for idx, (x_val, y_val) in enumerate(z_cpu):
            plt.annotate(
                str(idx), 
                (x_val, y_val), 
                textcoords="offset points", 
                xytext=(0, 8), 
                ha='center', 
                fontsize=9, 
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="orange")
            )
            
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        plt.grid(True, linestyle=":", alpha=0.6, zorder=2)
        plt.xlabel("Latent Dimension 1 (z_1)", fontsize=12, fontweight="bold")
        plt.ylabel("Latent Dimension 2 (z_2)", fontsize=12, fontweight="bold")
        plt.title("Sampled Latent Space Coordinates (Noise)", fontsize=14, fontweight="bold")
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        
        if show:
            plt.show()
        elif noise_filename is not None:
            parent_dir = os.path.dirname(noise_filename)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(noise_filename, bbox_inches="tight")
            plt.close()


#KL loss
def vae_gaussian_kl_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()


#reconstruction loss
def reconstruction_loss(x_reconstructed, x):
    bce_loss = nn.BCELoss()
    return bce_loss(x_reconstructed, x)


#total loss
def vae_loss(y_pred, y_true):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return 500 * recon_loss + kld_loss

