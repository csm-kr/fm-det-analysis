
from detectron2.data import transforms as T

def build_train_augmentation():
    return T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False),

        T.ResizeShortestEdge(short_edge_length=[400, 500, 600],
                                sample_style="choice", max_size=1333),

        T.RandomApply(
            T.RandomCrop(crop_type="absolute_range", crop_size=(384, 600)),
            prob=0.5
        ),

        T.ResizeShortestEdge(short_edge_length=(480, 512, 544, 576, 608, 640,
                                                672, 704, 736, 768, 800), 
                                                sample_style="choice", max_size=1333),

    ])

def build_test_augmentation():
    return T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=[800], sample_style="choice", max_size=1333),
    ])