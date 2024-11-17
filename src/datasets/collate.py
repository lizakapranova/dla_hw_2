import torch

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

    result_batch['mix_audio'] = pad_sequence([x['mix_audio'].squeeze(0) for x in dataset_items]).permute(1, 0)
    result_batch['speaker_audio'] = pad_sequence([x['speaker_audio'].squeeze(0) for x in dataset_items]).permute(1, 0)
    result_batch["video"] = None # FIX

    return result_batch
