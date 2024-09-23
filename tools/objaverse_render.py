import datasets
import pandas as pd
import os

from PIL import Image
import numpy as np

_VERSION = datasets.Version("0.0.2")
_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

class ObjaverseRender(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": [
                    "your/rendered/data/subsetA", # TODO: change the dataset directory as necessary
                    "your/rendered/data/subsetB", # TODO: change the dataset directory as necessary
                    ]
                    ,
                },
            ),
        ]

    def _generate_examples(self, data_dir):
        metadata_path = [os.path.join(x, 'train.jsonl') for x in data_dir]
        metadatas = [pd.read_json(x, lines=True) for x in metadata_path]

        for i, metadata in enumerate(metadatas):
            for _, row in metadata.iterrows():
                text = row["text"]

                image_path = row["image"]
                image_path = os.path.join(data_dir[i], image_path)
                image = open(image_path, "rb").read()

                conditioning_image_path = row["conditioning_image"]
                conditioning_image_path = os.path.join(
                    data_dir[i], row["conditioning_image"]
                )
                if not os.path.exists(conditioning_image_path):
                    dirname, basename = os.path.split(conditioning_image_path)
                    
                    cond1_path = os.path.join(dirname, basename.replace("cond", "m0r1"))
                    cond2_path = os.path.join(dirname, basename.replace("cond", "m5r5"))
                    cond3_path = os.path.join(dirname, basename.replace("cond", "m1r0"))
                    
                    img_cond1 = Image.open(cond1_path).convert('L')
                    img_cond2 = Image.open(cond2_path).convert('L')
                    img_cond3 = Image.open(cond3_path).convert('L')

                    # print(img_cond1.size, img_cond2.size, img_cond3.size)

                    img_cond = [img_cond1, img_cond2, img_cond3]
                    img_cond = [np.array(x) for x in img_cond]
                    img_cond = np.stack(img_cond, axis=-1)

                    Image.fromarray(img_cond).save(conditioning_image_path)
                    
                    
                conditioning_image = open(conditioning_image_path, "rb").read()

                yield row["image"], {
                    "text": text,
                    "image": {
                        "path": image_path,
                        "bytes": image,
                    },
                    "conditioning_image": {
                        "path": conditioning_image_path,
                        "bytes": conditioning_image,
                    },
                }