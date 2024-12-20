import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import cv2
from cv2 import dnn_superres

def pil_to_torch(img):
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return tensor

def torch_to_pil(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def upscale_bilinear(img_tensor, scale_factor):
    img_tensor = img_tensor.to('cuda')
    upsampled = F.interpolate(img_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    return upsampled.cpu()

def upscale_bicubic(img_tensor, scale_factor):
    img_tensor = img_tensor.to('cuda')
    upsampled = F.interpolate(img_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=True)
    return upsampled.cpu()

def upscale_lanczos(original_img, scale_factor):
    width, height = original_img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return original_img.resize((new_width, new_height), Image.LANCZOS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Upscaling Comparison (Bilinear, Bicubic, Lanczos, EDSR)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--scale_factor", type=float, default=2.0, choices=[2.0, 3.0, 4.0], help="Scaling factor for upscaling. (Supported: 2, 3, 4)")
    args = parser.parse_args()

    # 載入原始影像
    original_img = Image.open(args.input_image).convert("RGB")
    img_tensor = pil_to_torch(original_img)  # (1, C, H, W)

    # Bilinear Upscale on GPU
    bilinear_tensor = upscale_bilinear(img_tensor, args.scale_factor)
    bilinear_img = torch_to_pil(bilinear_tensor)
    bilinear_img.save("upscaled_bilinear.jpg")

    # Bicubic Upscale on GPU
    bicubic_tensor = upscale_bicubic(img_tensor, args.scale_factor)
    bicubic_img = torch_to_pil(bicubic_tensor)
    bicubic_img.save("upscaled_bicubic.jpg")

    # Lanczos Upscale on CPU
    lanczos_img = upscale_lanczos(original_img, args.scale_factor)
    lanczos_img.save("upscaled_lanczos.jpg")

    if args.scale_factor == 2.0:
        model_path = "./models/EDSR_x2.pb"
        model_scale = 2
    elif args.scale_factor == 3.0:
        model_path = "./models/EDSR_x3.pb"
        model_scale = 3
    elif args.scale_factor == 4.0:
        model_path = "./models/EDSR_x4.pb"
        model_scale = 4
    else:
        print("Error: Supported EDSR models are only for scale factors 2, 3, and 4.")
        exit(1)

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", model_scale)
    cv_img = cv2.imread(args.input_image)  # BGR format
    edsr_img = sr.upsample(cv_img)
    cv2.imwrite("upscaled_edsr.jpg", edsr_img)

    print("Upscaling completed. Results saved as:")
    print(" - upscaled_bilinear.jpg (GPU)")
    print(" - upscaled_bicubic.jpg (GPU)")
    print(" - upscaled_lanczos.jpg (CPU)")
    print(" - upscaled_edsr.jpg (CPU, EDSR_x{} model)".format(model_scale))
