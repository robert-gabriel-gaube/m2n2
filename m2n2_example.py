import cv2
import matplotlib.pyplot as plt

from src.m2n2_model import M2N2SegmentationModel
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator


def visualize(image_rgb, points, points_in_segment, segmentation):
    _, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    for p, c in zip(points, points_in_segment):
        axes[0].scatter([p[0]], [p[1]], color='green' if c else 'red')
    axes[0].imshow(image_rgb)
    axes[0].set_xlabel('Input Image')
    for p, c in zip(points, points_in_segment):
        axes[1].scatter([p[0]], [p[1]], color='green' if c else 'red')
    axes[1].imshow(segmentation.cpu().numpy(), cmap='gray')
    axes[1].set_xlabel('Segmentation')
    for p, c in zip(points, points_in_segment):
        axes[2].scatter([p[0]], [p[1]], color='green' if c else 'red')
    axes[2].imshow(image_rgb * segmentation.cpu().numpy().astype(int)[:, :, None])
    axes[2].set_xlabel('Cropped out segmentation')
    plt.show()


def main():
    # Inputs
    image_rgb = cv2.imread('images/image.jpg')[:, :, ::-1]
    points = [(300, 175), (135, 140), (200, 150), (200, 286)]
    points_in_segment = [True, True, True, False]

    # Predict
    attn_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')
    model = M2N2SegmentationModel(attn_aggregator)
    segmentation = model.segment(
        img=image_rgb,
        points=points,
        points_in_segment=points_in_segment
    )

    # Visualize Result
    visualize(image_rgb, points, points_in_segment, segmentation)


if __name__ == '__main__':
    main()
