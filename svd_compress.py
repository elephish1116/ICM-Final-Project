import argparse
import torch
import numpy as np
import math
import os
from PIL import Image

def compute_psnr(original, reconstructed, max_val=255.0):
    """
    Compute PSNR between the original and the reconstructed images.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10((max_val ** 2) / mse)
    return psnr

def select_k_energy(S, threshold=0.9):
    """
    Select k based on accumulated squared singular values.
    """
    total_energy = torch.sum(S ** 2)
    cumulative = 0.0
    for i in range(len(S)):
        cumulative += S[i].item() ** 2
        if cumulative / total_energy >= threshold:
            return i + 1
    return len(S)

def select_k_gap(S):
    """
    Select k based on the largest gap in the singular values.
    """
    max_gap = 0.0
    max_idx = 0
    for i in range(len(S) - 1):
        gap = S[i] - S[i + 1]
        if gap > max_gap:
            max_gap = gap
            max_idx = i
    return max_idx + 1

def svd_compress_torch(img_tensor, k):
    """
    Perform SVD compression on a single-channel tensor (H, W).
    Only keep the top k singular values.
    """
    U, S, Vh = torch.linalg.svd(img_tensor, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    compressed = U_k @ torch.diag(S_k) @ Vh_k
    return compressed

def select_k_quality_binary_search(channel, target_psnr):
    """
    Use binary search to find the smallest k such that PSNR >= target_psnr.
    If it's impossible, we return full rank (len(S)).
    
    channel: GPU tensor, shape (H, W), float
    target_psnr: desired PSNR in dB
    """
    # We do an SVD up front
    U, S, Vh = torch.linalg.svd(channel, full_matrices=False)
    max_k = len(S)

    # Move original channel to CPU for PSNR calculation
    channel_cpu = channel.cpu().numpy()

    # Define a helper function to reconstruct with k and compute PSNR
    def test_psnr(k_candidate):
        # Reconstruct on GPU
        U_k = U[:, :k_candidate]
        S_k = S[:k_candidate]
        Vh_k = Vh[:k_candidate, :]
        compressed_c = U_k @ torch.diag(S_k) @ Vh_k
        # Move to CPU for PSNR
        compressed_c_cpu = compressed_c.cpu().numpy()
        psnr_val = compute_psnr(channel_cpu, compressed_c_cpu, max_val=255)
        return psnr_val

    low, high = 1, max_k
    best_k = max_k
    # binary search
    while low <= high:
        mid = (low + high) // 2
        psnr_val = test_psnr(mid)
        if psnr_val >= target_psnr:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    return best_k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD Compression with Automatic Rank Selection")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--method", type=str, default="energy", choices=["energy", "gap", "quality"],
                        help="Method for selecting k: 'energy', 'gap', or 'quality'.")
    parser.add_argument("--energy_threshold", type=float, default=0.9,
                        help="Energy threshold for 'energy' method (default=0.9).")
    parser.add_argument("--target_psnr", type=float, default=30.0,
                        help="Target PSNR for 'quality' method (default=30 dB).")
    args = parser.parse_args()

    # Load the original image
    original_img = Image.open(args.input_image).convert("RGB")
    original_array = np.array(original_img, dtype=np.float32)

    # Convert to (C, H, W) PyTorch tensor
    img_tensor = torch.from_numpy(original_array).permute(2, 0, 1)

    # Optional: move to GPU
    img_tensor = img_tensor.to('cuda')

    compressed_channels = []
    for c in range(img_tensor.size(0)):
        channel = img_tensor[c, :, :].float()
        
        if args.method == "energy":
            # Just get S
            U, S, Vh = torch.linalg.svd(channel, full_matrices=False)
            k_c = select_k_energy(S, threshold=args.energy_threshold)
        elif args.method == "gap":
            U, S, Vh = torch.linalg.svd(channel, full_matrices=False)
            k_c = select_k_gap(S)
        else:  # "quality" with binary search
            k_c = select_k_quality_binary_search(channel, target_psnr=args.target_psnr)

        # Perform final reconstruction with k_c
        compressed_c = svd_compress_torch(channel, k_c)
        compressed_channels.append(compressed_c)

        print(f"Channel {c}: Selected k = {k_c}")

    # Combine channels back
    compressed_tensor = torch.stack(compressed_channels, dim=0)
    compressed_array = compressed_tensor.permute(1, 2, 0).cpu().numpy()
    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)

    # Compute overall PSNR
    psnr_compressed = compute_psnr(original_array, compressed_array, max_val=255)
    print("PSNR after SVD compression:", psnr_compressed, "dB")

    # Save output
    original_filename = os.path.basename(args.input_image)
    output_filename = "compressed_" + original_filename
    Image.fromarray(compressed_array).save(output_filename)
    print("Saved '" + output_filename + "'.")
