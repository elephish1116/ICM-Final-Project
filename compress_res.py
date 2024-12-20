import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def svd_compress_torch(img_tensor, k):
    """
    Perform SVD compression on a single-channel tensor (H, W) and reconstruct it using only the top k singular values.
    Returns the reconstructed tensor (H, W).
    """
    U, S, Vh = torch.linalg.svd(img_tensor, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    compressed = U_k @ torch.diag(S_k) @ Vh_k
    return compressed

def compute_psnr(original, reconstructed, max_val=255.0):
    """
    Compute PSNR between the original and reconstructed images.
    original, reconstructed: numpy arrays with shape (H, W, C).
    max_val: 255 for 8-bit images.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val**2) / mse)
    return psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD Compression and Interpolation")
    parser.add_argument("--mode", type=str, default="bilinear", choices=["bilinear", "bicubic"], help="Choose the interpolation mode: 'bilinear' or 'bicubic'.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Load the original image and convert it to a tensor (C, H, W)
    original_img = Image.open(args.input).convert("RGB")
    original_array = np.array(original_img, dtype=np.float32)
    img_tensor = torch.from_numpy(original_array)  # (H, W, C)
    img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)

    img_tensor = img_tensor.to('cuda')

    # Set the SVD compression rank k
    k = 300

    compressed_channels = []
    for c in range(img_tensor.size(0)):
        channel = img_tensor[c, :, :]  # (H, W)
        compressed_c = svd_compress_torch(channel, k)
        compressed_channels.append(compressed_c)

    # Combine the compressed channels back (C, H, W)
    compressed_tensor = torch.stack(compressed_channels, dim=0)

    # Convert the compressed result back to (H, W, C)
    compressed_array = compressed_tensor.permute(1, 2, 0).cpu().numpy()
    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)

    # Compute PSNR between compressed and original images
    psnr_compressed = compute_psnr(original_array, compressed_array, max_val=255)
    print("PSNR after SVD compression:", psnr_compressed, "dB")

    # Use the chosen interpolation method to upscale the image
    compressed_tensor = compressed_tensor.unsqueeze(0)  # (1, C, H, W)
    scale_factor = 2.0

    resized_tensor = F.interpolate(compressed_tensor, scale_factor=scale_factor, mode=args.mode, align_corners=True)
    resized_tensor = resized_tensor.squeeze(0)  # (C, newH, newW)

    resized_array = resized_tensor.permute(1, 2, 0).cpu().numpy()
    resized_array = np.clip(resized_array, 0, 255).astype(np.uint8)

    # Compute PSNR after resizing by comparing with a similarly resized original image
    # Here we use Pillow to resize the original image with the same method for comparison
    pil_mode = Image.BICUBIC if args.mode == "bicubic" else Image.BILINEAR
    original_resized = np.array(original_img.resize((resized_array.shape[1], resized_array.shape[0]), pil_mode), dtype=np.uint8)
    psnr_resized = compute_psnr(original_resized, resized_array, max_val=255)
    print(f"PSNR after resizing ({args.mode}):", psnr_resized, "dB")

    # Display results
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_array.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(compressed_array)
    axes[1].set_title(f"Compressed (k={k})\nPSNR: {psnr_compressed:.2f} dB")
    axes[1].axis('off')

    axes[2].imshow(resized_array)
    axes[2].set_title(f"Resized ({args.mode}, x{scale_factor})\nPSNR: {psnr_resized:.2f} dB")
    axes[2].axis('off')
    '''

    
    Image.fromarray(compressed_array).save("compressed_image.jpg")
    Image.fromarray(resized_array).save(f"resized_{args.mode}_from_svd_compressed.jpg")
