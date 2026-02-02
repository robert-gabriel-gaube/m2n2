import argparse
import json
import os
import shutil
from collections import defaultdict
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import cv2
import numpy as np
from pycocotools.mask import encode as rle_encode
from tqdm import tqdm


def get_bbox_from_binary_seg(seg):
    x_slices = np.max(seg, axis=0)
    x = int(np.argmax(x_slices))
    width = int(len(x_slices) - np.argmax(x_slices[::-1]) - x)

    y_slices = np.max(seg, axis=1)
    y = int(np.argmax(y_slices))
    height = int(len(y_slices) - np.argmax(y_slices[::-1]) - y)

    return [x, y, width, height]


def download_and_extract_zip(download_url=None, target_download_directory=None):
    if download_url is not None and target_download_directory is not None:
        print('Downloading: ', download_url)
        http = urlopen(download_url)
        zipfile = ZipFile(BytesIO(http.read()))
        os.makedirs(target_download_directory, exist_ok=True)
        print('Extracting to ', target_download_directory)
        zipfile.extractall(path=target_download_directory)


def collect_samples_from_folders(rgb_folder, seg_folder, id_prefix=''):
    # Collecting images and annotations
    samples = defaultdict(dict)
    for fname in os.listdir(rgb_folder):
        img_id, dtype = fname.split('.')
        img_id = id_prefix + img_id
        samples[img_id]['rgb'] = os.path.join(rgb_folder, fname)
    for fname in os.listdir(seg_folder):
        img_id, dtype = fname.split('.')
        img_id = id_prefix + img_id
        samples[img_id]['seg'] = os.path.join(seg_folder, fname)
    return samples


def create_coco_one_segment_per_image_dataset(samples: dict,
                                              out_dataset_folder,
                                              description='FBRS-Type Dataset',
                                              use_image_id_as_new_image_filename=False):
    """
    :param samples: dictionary mapping from image id to filepaths:
        samples[id]['rgb'] = full filepath to rgb image
        samples[id]['seg'] = full filepath to binary segmentation file (everything except (0, 0, 0) is treated as foreground)
    :param description:
    :param out_dataset_folder:
    :return:
    """
    # -----------
    # Some checks
    # -----------
    # Every id has seg and rgb
    for v in samples.values():
        if 'seg' not in v or 'rgb' not in v:
            raise Exception('Missing either rgb or seg entry for:' + str(v))

    # every filename is unique
    assert len(set(s['seg'] for s in samples.values())) == len(samples)
    assert len(set(s['rgb'] for s in samples.values())) == len(samples)

    # ----
    # Info
    # ----
    out = dict()
    out['info'] = {
        'description': description
    }

    # ------
    # Images
    # ------
    out_img_dir = os.path.join(out_dataset_folder, 'images')
    out_fname = os.path.join(out_dataset_folder, 'gt.json')
    img_ids = list(sorted(samples.keys()))
    out['images'] = []
    os.makedirs(out_img_dir, exist_ok=True)
    for img_id in tqdm(img_ids, f'Copying images of {description}'):

        src_filepath = samples[img_id]['rgb']
        dst_basename = os.path.basename(samples[img_id]['rgb'])
        if use_image_id_as_new_image_filename:
            dst_basename = f'{img_id}.{dst_basename.split(".")[-1]}'
        dst_filepath = os.path.join(out_img_dir, dst_basename)
        if not os.path.exists(src_filepath):
            print('Could not find rgb image file:', src_filepath, 'for id', img_id)
            exit()
        shutil.copyfile(src_filepath, dst_filepath)
        img = cv2.imread(src_filepath)
        out['images'].append({
            'id': img_id,
            'width': int(img.shape[1]),
            'height': int(img.shape[0]),
            'file_name': dst_basename
        })

    # ----------
    # Categories
    # ----------
    # We only have one category, the foreground. There could also be a background category, but we do not predict this here.
    out['categories'] = [{
        'id': 0,
        'name': 'foreground',
        'supercategory': 0
    }]

    # -----------
    # Annotations
    # -----------
    out['annotations'] = []
    for img_id in tqdm(img_ids, f'Converting Annotation of {description}'):
        ann_id = img_id + '_foreground'
        fname = samples[img_id]['seg']
        if not os.path.exists(fname):
            print('Could not find segmentation file:', fname)
            exit()
        seg = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        mask = (np.max(seg, axis=2) > 0).astype(int) if len(seg.shape) > 2 else (seg > 0).astype(int)
        rle = rle_encode(np.asfortranarray(mask.astype(np.uint8)))
        out['annotations'].append({
            'id': ann_id,
            'image_id': img_id,
            'category_id': 0,  # Zero for foreground as we defined earlier
            'segmentation': {
                'size': list(rle['size']),
                'counts': rle['counts'].decode('ascii')  # Based on this: https://github.com/cocodataset/cocoapi/issues/70
            },
            'area': int(np.sum(mask)),
            'bbox': get_bbox_from_binary_seg(mask),
            'iscrowd': 0  # We have purely segmentation, so we set it to zero always
        })

    os.makedirs(os.path.split(out_fname)[0], exist_ok=True)
    with open(out_fname, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4)
    print('Created file:', out_fname)


def convert_fbrs_dataset(images_folder,
                         segments_folder,
                         out_dataset_folder,
                         description='FBRS-Type Dataset',
                         download_url=None,
                         target_download_directory=None):
    # Download data
    download_and_extract_zip(download_url, target_download_directory)

    # Collect samples
    samples = collect_samples_from_folders(
        rgb_folder=images_folder,
        seg_folder=segments_folder
    )

    # Create dataset
    create_coco_one_segment_per_image_dataset(
        samples=samples,
        out_dataset_folder=out_dataset_folder,
        description=description
    )


def convert_sbd_dataset(download_torchvision_dataset, image_set='val', out_dataset_folder='/datasets/fbrs_sbd', download=False):
    from torchvision.datasets.sbd import SBDataset

    out_img_dir = os.path.join(out_dataset_folder, 'images')
    out_fname = os.path.join(out_dataset_folder, f'{image_set}.json')

    sbd_dataset = SBDataset(root=download_torchvision_dataset, image_set=image_set, download=download, mode='segmentation')

    # ----
    # Info
    # ----
    out = dict()
    out['info'] = {
        'description': 'SBD Instance Segmentation Dataset'
    }

    # ----------
    # Categories
    # ----------
    # We only have instance ids, so we simply say instance as only category
    out['categories'] = [{
        'id': 0,
        'name': 'instance',
        'supercategory': 0
    }]

    out['images'] = []
    out['annotations'] = []
    os.makedirs(out_img_dir, exist_ok=True)
    ann_idx = 0
    for img_id, (image, instances_mask) in enumerate(tqdm(sbd_dataset, f'Creating SBD {image_set} instance segmentation dataset')):
        image = np.array(image)
        instances_mask = np.array(instances_mask)

        # ------
        # Images
        # ------
        img_dst_filepath = os.path.join(out_img_dir, f'{img_id}.png')
        cv2.imwrite(img_dst_filepath, image[:, :, ::-1])    # need to convert rgb to bgr
        out['images'].append({
            'id': img_id,
            'width': int(image.shape[1]),
            'height': int(image.shape[0]),
            'file_name': f'{img_id}.png'
        })

        # -----------
        # Annotations
        # -----------
        instance_ids = np.unique(instances_mask)
        for instance_id in instance_ids:
            if instance_id == 0:
                # We do not want background as an instance
                continue

            ann_id = ann_idx
            mask = (instances_mask == instance_id).astype(int)
            rle = rle_encode(np.asfortranarray(mask.astype(np.uint8)))
            out['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 0,  # Zero for instance as we defined earlier
                'segmentation': {
                    'size': list(rle['size']),
                    'counts': rle['counts'].decode('ascii')  # Based on this: https://github.com/cocodataset/cocoapi/issues/70
                },
                'area': int(np.sum(mask)),
                'bbox': get_bbox_from_binary_seg(mask),
                'iscrowd': 0  # We have purely segmentation, so we set it to zero always
            })
            ann_idx += 1

    os.makedirs(os.path.split(out_fname)[0], exist_ok=True)
    with open(out_fname, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4)
    print('Created file:', out_fname)


def prepare_fbrs_datasets(grabcut=True, berkeley=True, sbd=True, davis=True, brats=False, oaizib=False):
    # Based on this repository: https://github.com/SamsungLabs/fbrs_interactive_segmentation?tab=readme-ov-file
    if grabcut:
        convert_fbrs_dataset(
            images_folder='/content/datasets/raw/GrabCut/data_GT',
            segments_folder='/content/datasets/raw/GrabCut/boundary_GT',
            out_dataset_folder='./datasets/coco/GrabCut',
            description='FBRS Version of GrabCut Dataset',
            download_url=None,
            target_download_directory=None
        )

    if berkeley:
        convert_fbrs_dataset(
            images_folder='/content/datasets/raw/Berkeley/images',
            segments_folder='/content/datasets/raw/Berkeley/masks',
            out_dataset_folder='./datasets/coco/Berkeley',
            description='FBRS Version of Berkeley Dataset',
            download_url=None,
            target_download_directory=None
        )

    if davis:
        convert_fbrs_dataset(
            images_folder='/content/datasets/raw/DAVIS/img',
            segments_folder='/content/datasets/raw/DAVIS/gt',
            out_dataset_folder='./datasets/coco/DAVIS',
            description='FBRS Version of DAVIS Dataset',
            download_url=None,
            target_download_directory=None
        )

    if sbd:
        convert_sbd_dataset(
            download_torchvision_dataset='/content/datasets/raw/torchvision_sbd',
            out_dataset_folder='./datasets/coco/SBD',
            image_set='val',     # or 'train'
            download=True
        )

    if brats:
        convert_fbrs_dataset(
            download_url='https://drive.google.com/uc?export=download&id=1FQR_VCee_FdVk6U-rDZPdDKF5N8qwyql',    # Link created with this website https://sites.google.com/site/gdocs2direct/ and taken from SimpleClick Repository
            target_download_directory='./datasets/raw',
            images_folder='./datasets/raw/BraTS20/image',
            segments_folder='./datasets/raw/BraTS20/annotation',
            out_dataset_folder='./datasets/coco/BraTS20',
            description='BraTS20 Dataset'
        )

    if oaizib:
        # Can not download directly because it is too large for Google Drive virus scan and it opens an extra page
        # Therefore please download the dataset under https://drive.google.com/uc?export=download&id=1jmXUGCUFu-qND48c8RndrUqyWxRGv0cg manually and extract the folder in
        # ./datasets/raw/OIA-ZIB
        convert_fbrs_dataset(
            # download_url='https://drive.google.com/uc?export=download&id=1jmXUGCUFu-qND48c8RndrUqyWxRGv0cg',
            # target_download_directory='./datasets/raw',
            images_folder='./datasets/raw/OAI-ZIB/test/image',
            segments_folder='./datasets/raw/OAI-ZIB/test/annotations',
            out_dataset_folder='./datasets/coco/OAI-ZIB',
            description='OAI-ZIB Dataset'
        )


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation")
    parser.add_argument('--grabcut', action='store_true', default=False)
    parser.add_argument('--berkeley', action='store_true', default=False)
    parser.add_argument('--sbd', action='store_true', default=False)
    parser.add_argument('--davis', action='store_true', default=False)
    parser.add_argument('--brats', action='store_true', default=False)
    parser.add_argument('--oaizib', action='store_true', default=False)
    args = parser.parse_args()
    prepare_fbrs_datasets(**vars(args))


if __name__ == '__main__':
    main()
