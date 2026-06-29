import os
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


DATASET_NAME = "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
OUTPUT_ROOT = Path("brain_tumor_unet")


def main():
    dataset_path = Path(kagglehub.dataset_download(DATASET_NAME))
    print("Dataset downloaded to:", dataset_path)

    for split in ["train", "valid", "test"]:
        split_dir = dataset_path / split
        ann_file = split_dir / "_annotations.coco.json"

        img_out = OUTPUT_ROOT / split / "imgs"
        mask_out = OUTPUT_ROOT / split / "masks"

        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        coco = COCO(str(ann_file))

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]

            img_path = split_dir / file_name
            image = Image.open(img_path).convert("RGB")
            image.save(img_out / file_name)

            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                mask = np.maximum(mask, coco.annToMask(ann) * 255)

            mask_name = Path(file_name).stem + ".png"
            Image.fromarray(mask).save(mask_out / mask_name)

    print("Done converting COCO annotations to U-Net masks.")


if __name__ == "__main__":
    main()
