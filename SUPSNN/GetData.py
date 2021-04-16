import os
import random
import numpy as np
import joblib
import logging

logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)


class GetData:
    def __init__(self, data_dir):
        img_list = []
        labels_list = []
        self.source_list = []

        examples = 0
        logging.info('Loading images...')
        label_dir = os.path.join(data_dir, "Labels")
        img_dir = os.path.join(data_dir, "Spikes")

        for label_root, dir, files in os.walk(label_dir):
            for file in files:
                if not file.endswith((".pkl")):
                    continue
                try:
                    folder = os.path.relpath(label_root, label_dir)
                    img_root = os.path.join(img_dir, folder)
                    img = joblib.load(os.path.join(img_root, file))
                    label = joblib.load(os.path.join(label_root, file))

                    img_list.append((np.array(img)).astype(np.float32))
                    labels_list.append((np.array(label)).astype(np.float32))
                    examples = examples + 1
                except Exception as e:
                    print(e)

        self.examples = examples
        logging.info("Number of samples found: " + str(examples))
        self.img = np.array(img_list)
        self.labels = np.array(labels_list)

    def next_batch(self, batch_size):

        if len(self.source_list) < batch_size:
            new_source = list(range(self.examples))
            random.shuffle(new_source)
            self.source_list.extend(new_source)

        examples_idx = self.source_list[:batch_size]
        del self.source_list[:batch_size]

        return self.img[examples_idx, ...], self.labels[examples_idx, ...]
