import torch
from torch import stack

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["speaker_folder"] = [x["speaker_folder"] for x in dataset_items]
    result_batch["mix_name"] = [x["mix_name"] for x in dataset_items]

    result_batch["mix_audio"] = pad_sequence([x["mix_audio"].squeeze(0) for x in dataset_items]).permute(1, 0)
    for item in dataset_items:
        if item["speaker_audio"] is None:
            result_batch["speaker_audio"] = None
            break
    else:
        result_batch["speaker_audio"] = pad_sequence([x["speaker_audio"].squeeze(0) for x in dataset_items]).permute(1, 0)
    result_batch["video"] = stack([x["video"] for x in dataset_items])

    return result_batch
