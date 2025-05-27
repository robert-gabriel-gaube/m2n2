import matplotlib.pyplot as plt
import cv2
import torch

from src.markov_map import create_markov_map_from_point
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
from src.utils import change_temperature, matrix_ipf


def adjust_temperature_and_ipf(attention: torch.Tensor,
                               temperature=0.65):
    bh, bw, h, w = attention.shape

    # Change temperature and apply IPF
    attn = change_temperature(attention.reshape(bh * bw, h * w), temperature=temperature).reshape(bh, bw, h, w)
    attn = matrix_ipf(attn.reshape(bh * bw, h * w), iterations=200).reshape(bh, bw, h, w)
    return attn


def visualize(image_rgb, point, markov_map):
    _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].scatter([point[0]], [point[1]], label='Input Point')
    axes[0].imshow(image_rgb)
    axes[0].set_xlabel('Input Image')
    axes[0].legend()
    axes[1].scatter([point[0]], [point[1]], label='Input Point')
    axes[1].imshow(-markov_map.cpu().numpy(), cmap='gray')
    axes[1].set_xlabel('Markov-map')
    axes[1].legend()
    plt.show()


def main():
    # Inputs
    image_rgb = cv2.imread('./images/image.jpg')[:, :, ::-1]
    point = (300, 175)

    # Get attention tensor
    attn_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')
    attention_tensor = attn_aggregator.extract_attention(image_rgb)

    # Generate Markov-map
    A = adjust_temperature_and_ipf(attention_tensor)
    markov_map = create_markov_map_from_point(
        image=image_rgb,            # RGB input image as numpy array of shape (Height, Width, 3)
        A_tensor=A,                 # Attention tensor of shape (R, R, R, R)
        point=point,                # 2D point in image pixel coordinates: (x, y)
    )

    # Visualize Result
    visualize(image_rgb, point, markov_map)


if __name__ == '__main__':
    main()
