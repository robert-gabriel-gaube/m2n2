import argparse

import cv2
import numpy as np

from src.m2n2_model import M2N2SegmentationModel
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator


class DemoApp(object):
    def __init__(self, img, segmentor, **kwargs):
        super().__init__(**kwargs)
        self.window_name = 'M2N2SegmentationModel Demo'
        self.cursor = (0, 0)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.cursor_update)
        self.segmentor = segmentor
        self.img = img
        self.segmentation = None
        self.segmentation_soft = None
        self.points = []
        self.points_in_segment = []
        self.render_output = None
        self.redraw()

    def start(self):
        while True:
            cv2.imshow(self.window_name, self.draw())
            k = cv2.waitKey(25)
            if k == 27:
                break
            self.key_callback(k)
        cv2.destroyAllWindows()

    def cursor_update(self, event, x, y, flags, param):
        self.cursor = x, y

    def key_callback(self, k):
        key_i = 105
        key_o = 111
        key_r = 114
        img_coord = np.array(self.cursor)
        if k == key_i or k == key_o:
            self.points.append(img_coord)
            self.points_in_segment.append(k == key_i)
            self.update_prediction()
        elif k == key_r:
            self.points = []
            self.points_in_segment = []
            self.update_prediction()

    def update_prediction(self):
        if len(self.points) <= 0:
            self.segmentation = None
            self.redraw()
            return False

        self.segmentation = self.segmentor.segment(
            img=self.img,
            points=self.points,
            points_in_segment=self.points_in_segment
        )
        self.segmentation = self.segmentation.cpu().numpy()
        self.redraw()

    def redraw(self):
        out = self.img.copy()
        if self.segmentation is not None:
            out = ((out - (out * self.segmentation[:, :, None]) // 2) + np.array([[[30, 100, 30]]]) * self.segmentation[:, :, None]).astype(np.uint8)
        for i, (coords, in_segment) in enumerate(zip(self.points, self.points_in_segment)):
            c = (int(coords[0] + 0.5), int(coords[1] + 0.5))
            cv2.drawMarker(out, c, (10, 10, 10), cv2.MARKER_TILTED_CROSS, 10, 5)
            cv2.drawMarker(out, c, (150, 150, 150) if not in_segment else (10, 200, 10), cv2.MARKER_TILTED_CROSS, 10, 3)
        self.render_output = out.copy()

    def draw(self) -> np.ndarray:
        return self.render_output[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser(description="M2N2 Interactive Demo")
    parser.add_argument('filename')
    args = parser.parse_args()
    print("""Controls:
    Key I: Insert foreground keypoint at cursor location
    Key O: Insert background keypoint at cursor location
    Key R: Reset segmentation and delete all key points
    Key ESC: Close the demo
    """)

    # Inputs
    image_rgb = cv2.imread(args.filename)[:, :, ::-1]

    # Create Interactive Demo
    attn_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')
    model = M2N2SegmentationModel(attn_aggregator)
    DemoApp(
        img=image_rgb,
        segmentor=model
    ).start()


if __name__ == '__main__':
    main()
