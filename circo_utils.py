import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal

import numpy as np
import PIL
import PIL.Image
import torch.utils.data
from torch.utils.data import Dataset

class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'], mode: Literal['relative', 'classic'],
                 preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def get_semantic_aspects(self, index):
        """ Returns the semantic aspects for a given query"""
        return self.annotations[index].get('semantic_aspects', [])

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path).convert('RGB'))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path).convert('RGB'))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path).convert('RGB'))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


def compute_metrics(data_path: Path, predictions_dict: Dict[int, List[int]], ranks: List[int]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[str, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]: Dictionaries with the AP and Recall for each rank,
            and the semantic mAP@10 for each semantic aspect
    """

    relative_val_dataset = CIRCODataset(data_path, split='val', mode='relative', preprocess=lambda x: x)

    semantic_aspects_list = ['cardinality', 'addition', 'negation', 'direct_addressing', 'compare_change',
                              'comparative_statement', 'statement_with_conjunction', 'spatial_relations_background',
                              'viewpoint']

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)
    semantic_aps_at10 = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = relative_val_dataset.get_target_img_ids(int(query_id))
        semantic_aspects = relative_val_dataset.get_semantic_aspects(int(query_id))
        gt_img_ids = target['gt_img_ids']
        target_img_id = target['target_img_id']

        # Check if the predictions are unique
        if len(set(predictions)) != len(predictions):
            raise ValueError(f"Query {query_id} has duplicate predictions. Please ensure to provide unique predictions"
                             f"for each query.")

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

        # Compute the AP@10 for each semantic aspect
        for aspect in semantic_aspects:
            semantic_aps_at10[aspect].append(float(np.sum(precisions[:10]) / min(len(gt_img_ids), 10)))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    map_atk = {}
    recall_atk = {}
    semantic_map_at10 = {}
    for rank in ranks:
        map_atk[rank] = float(np.mean(aps_atk[rank]))
        recall_atk[rank] = float(np.mean(recalls_atk[rank]))

    # Compute the mean AP@10 for each semantic aspect and store them in a dictionary
    for aspect in semantic_aspects_list:
        semantic_map_at10[aspect] = float(np.mean(semantic_aps_at10[aspect]))

    return map_atk, recall_atk, semantic_map_at10
