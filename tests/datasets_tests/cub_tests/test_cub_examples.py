import unittest

from chainer import testing

from chainercv.datasets import CUBLabelDataset
from chainercv.datasets import CUBKeypointDataset


EXAMPLE_COUNT = {
    'train': 5994,
    'test': 5794,
    'traintest': 5994+5794
}


TRAIN_EXAMPLE = {
    'image_id': 3930,
    'path':
    '068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0119_57575.jpg',
    'class_id': 68,
    'bbox_xywh': [159.0, 232.0, 114.0, 113.0],
    'visible': [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    'point_xy': [[203.0, 275.0], [256.0, 248.0], [238.0, 298.0],
                 [241.0, 281.0], [230.0, 237.0], [238.0, 241.0],
                 [0.0, 0.0], [237.0, 315.0], [0.0, 0.0],
                 [214.0, 263.0], [225.0, 247.0], [212.0, 321.0],
                 [201.0, 296.0], [169.0, 339.0], [243.0, 257.0]]}
TEST_EXAMPLE = {
    'image_id': 7859,
    'path': '134.Cape_Glossy_Starling/Cape_Glossy_Starling_0081_129220.jpg',
    'class_id': 134,
    'bbox_xywh': [223.0, 50.0, 268.0, 198.0],
    'visible': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    'point_xy': [[344.0, 90.0], [236.0, 83.0], [320.0, 162.0],
                 [276.0, 126.0], [268.0, 63.0], [251.0, 73.0],
                 [260.0, 77.0], [322.0, 191.0], [342.0, 119.0],
                 [306.0, 64.0], [0.0, 0.0], [349.0, 203.0],
                 [0.0, 0.0], [458.0, 185.0], [260.0, 97.0]]}


@testing.parameterize(*testing.product({
    'split': ['train', 'test', 'traintest'],
}))
class TestCUBLabelDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CUBLabelDataset(split=self.split, return_bbox=True)

    def test_example_count(self):
        assert len(self.dataset) == EXAMPLE_COUNT[self.split]

    def _test_example_aux(self, example, positive):
        path = example['path']
        if not positive:
            assert path not in self.dataset.paths
            return

        assert path in self.dataset.paths

        index = self.dataset.paths.index(path)
        # assert index == example['image_id'] - 1

        image, label, bbox = self.dataset[index]
        assert label == example['class_id'] - 1

        x, y, w, h = example['bbox_xywh']
        assert (bbox == (y, x, y+h, x+w)).all()

    def test_cub_train_example(self):
        print('train', self.split)
        self._test_example_aux(TRAIN_EXAMPLE, 'train' in self.split)

    def test_cub_test_example(self):
        print('test', self.split)
        self._test_example_aux(TEST_EXAMPLE, 'test' in self.split)


@testing.parameterize(*testing.product({
    'split': ['train', 'test', 'traintest'],
}))
class TestCUBKeypointDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CUBKeypointDataset(split=self.split, return_bbox=True)

    def test_example_count(self):
        assert len(self.dataset) == EXAMPLE_COUNT[self.split]

    def _test_example_aux(self, example, positive):
        path = example['path']
        if not positive:
            assert path not in self.dataset.paths
            return

        assert path in self.dataset.paths

        index = self.dataset.paths.index(path)
        # assert index == example['image_id'] - 1

        image, point, visible, bbox = self.dataset[index]
        assert (point[0, :, ::-1] == example['point_xy']).all()
        assert (visible == example['visible']).all()

        x, y, w, h = example['bbox_xywh']
        assert (bbox == (y, x, y+h, x+w)).all()

    def test_cub_train_example(self):
        print('train', self.split)
        self._test_example_aux(TRAIN_EXAMPLE, 'train' in self.split)

    def test_cub_test_example(self):
        print('test', self.split)
        self._test_example_aux(TEST_EXAMPLE, 'test' in self.split)
