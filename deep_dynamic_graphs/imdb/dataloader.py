from __future__ import print_function, division

import os
import numpy as np


class IMDBDataLoader(object):
    """Helper to load batches of imdb reviews and sentiment target.

    See: http://ai.stanford.edu/~amaas/data/sentiment/ for details and download
    data.

    :param dataset_path: path to folder with sub folders 'pos' and 'neg' with
        text files on format <id>_<score>.txt
    :type dataset_path: str

    :param label: 'binary' or 'score' (0-9) labels
    :type label: str
    """
    def __init__(self, dataset_path, label='binary'):
        if not label == 'binary':
            raise ValueError('only supports binary labels for now')
        self.dataset_path = dataset_path
        self.positives_index = self._get_index_from_dir('pos')
        self.negatives_index = self._get_index_from_dir('neg')
        self.index = self.negatives_index + self.positives_index

    def get_epoch_iterator(self, batch_size):
        """For iteration over one epoch with given batch size"""
        load_idx = np.random.permutation(len(self.index))
        for batch_start in range(0, len(self.index), batch_size):
            batch_idx = load_idx[batch_start:batch_start + batch_size]
            input_target_s = [
                (self._load_text(self.index[i][0]), [self.index[i][2]])
                for i in batch_idx
            ]
            yield input_target_s

    def _get_index_from_dir(self, which_sentiment):
        filepaths = [
            os.path.join(which_sentiment, filename) for
            filename in os.listdir(
                os.path.join(self.dataset_path, which_sentiment)
            )
        ]
        ratings = [int(filename[:-4].split('_')[-1]) for filename in filepaths]
        binary_label = [int(rating > 5) for rating in ratings]
        return zip(filepaths, ratings, binary_label)

    def _load_text(self, filename):
        with open(os.path.join(self.dataset_path, filename)) as f:
            return f.read()
