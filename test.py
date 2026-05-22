# test.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import matplotlib
matplotlib.use("agg")
from torchvision import datasets, transforms
from pyimagesearch import config, network, utils

def main():
    print("[INFO] Thiết lập transform và tải dữ liệu test...")
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    testset = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True)

    print("[INFO] Khởi tạo mô hình VAE...")
    encoder = network.Encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM).to(config.DEVICE)
    decoder = network.Decoder(config.EMBEDDING_DIM, config.SHAPE_BEFORE_FLATTENING).to(config.DEVICE)
    vae = network.VAE(encoder, decoder).to(config.DEVICE)

    if not os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"[LỖI] Không tìm thấy file weights tại {config.MODEL_WEIGHTS_PATH}!")
        print("Vui lòng chạy 'python train.py' để train mô hình trước khi visualize.")
        return

    print(f"[INFO] Đang tải trọng số (weights) từ {config.MODEL_WEIGHTS_PATH}...")
    checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.DEVICE)
    vae.load_state_dict(checkpoint["vae"])
    
    # Thiết lập mô hình ở chế độ evaluation
    vae.eval()
    encoder.eval()
    decoder.eval()

    print("[INFO] Visualize: Real vs Reconstructed Images...")
    utils.display_random_images(
        test_loader, 
        vae,
        file_recon=config.FILE_RECON_AFTER_TRAINING, 
        file_real=config.FILE_REAL_AFTER_TRAINING,
        title_recon="Reconstructed Images",
        title_real="Real Images",
        display_real=True,
        num_images=32,
        num_images_per_row=8
    )

    print(f"[INFO] Plotting Latent Space lưu tại {config.LATENT_SPACE_PLOT}...")
    utils.plot_latent_space(test_loader, encoder, show=False)

    print(f"[INFO] Plotting Image Grid on Embeddings lưu tại {config.IMAGE_GRID_EMBEDDINGS_PLOT}...")
    utils.plot_image_grid_on_embeddings(test_loader, encoder, decoder, show=False)

    print(f"[INFO] Plotting Linearly Sampled Reconstructions lưu tại {config.LINEARLY_SAMPLED_RECONSTRUCTIONS_PLOT}...")
    utils.plot_linearly_sampled_reconstructions(decoder, show=False)

    print(f"[INFO] Plotting Normally Sampled Reconstructions lưu tại {config.NORMALLY_SAMPLED_RECONSTRUCTIONS_PLOT}...")
    utils.plot_normally_sampled_reconstructions(decoder, show=False)

    print(f"[INFO] Plotting Latent Space Histogram lưu tại {config.LATENT_SPACE_HISTOGRAM_PLOT}...")
    utils.plot_latent_space_histogram(test_loader, encoder, show=False)

    print(f"[INFO] Plotting Random Generated Images & Noise lưu tại {config.RANDOM_GENERATED_IMAGES_PLOT} và {config.RANDOM_GENERATED_NOISE_PLOT}...")
    utils.random_generating_images(
        decoder, 
        filename=config.RANDOM_GENERATED_IMAGES_PLOT, 
        noise_filename=config.RANDOM_GENERATED_NOISE_PLOT, 
        show=False
    )

    print("[INFO] Hoàn tất! Tất cả các file ảnh đã được lưu trong thư mục 'output/'.")

if __name__ == "__main__":
    main()
