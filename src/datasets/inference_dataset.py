import os
import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm
import torchaudio
import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from torch.utils.data import Dataset


class InferenceDataset(BaseDataset):
    def __init__(self, dataset_path, *args, **kwargs):
        self._data_dir = ROOT_PATH / dataset_path
        index = self._get_or_create_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_create_index(self):
        index_path = self._data_dir / 'index.json'
        if index_path.exists():
            return read_json(str(index_path))
        index = []
        audio_path = self._data_dir / 'audio'
        video_path = self._data_dir / 'mouths'
        for mix_audio in tqdm(os.listdir(audio_path / 'mix')):
            s1 = mix_audio[:mix_audio.find('_')]
            s2 = mix_audio[mix_audio.find('_') + 1: mix_audio.find('.')]
            mix_name = mix_audio[:mix_audio.find('.')]

            index.extend([
                {
                    'mix_path': str(audio_path / 'mix' / mix_audio),
                    'audio_path': str(audio_path / 's1' / mix_audio),
                    'video_path': str(video_path / f"{s1}.npz"),
                    'speaker_folder': 's1',
                    'mix_name': mix_name
                },
                {
                    'mix_path': str(audio_path / 'mix' / mix_audio),
                    'audio_path': str(audio_path / 's2' / mix_audio),
                    'video_path': str(video_path / f"{s2}.npz"),
                    'speaker_folder': 's2',
                    'mix_name': mix_name
                }
            ])

        write_json(index, index_path)
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "mix_path" in entry, (
                "Each dataset item should include field 'mix_path'" " - path to mix audio file."
            )
            assert "audio_path" in entry, (
                "Each dataset item should include field 'audio_path'" " - path to ground-truth speaker audio file."
            )
            assert "video_path" in entry, (
                "Each dataset item should include field 'video_path'" " - path to video file."
            )

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        mix_path = data_dict['mix_path']
        audio_path = data_dict['audio_path']
        video_path = data_dict['video_path']
        speaker_folder = data_dict['speaker_folder']
        mix_name = data_dict['mix_name']

        mix = self.load_audio(mix_path)
        speaker_audio = self.load_audio(audio_path)
        video = torch.from_numpy(np.load(video_path)['data']).float() / 255
        
        _, h, w = video.shape
        th, tw = (88, 88)
        delta_w = int(round((w - tw)) / 2.)
        delta_h = int(round((h - th)) / 2.)
        video = video[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        (mean, std) = (0.421, 0.165)
        video =  (video - mean) / std

        instance_data = {'mix_audio': mix, 'speaker_audio': speaker_audio, 'video': video, 
                         'speaker_folder': speaker_folder, 'mix_name': mix_name}
        return instance_data


    @staticmethod
    def load_audio(path):
        # assert Path(path).exists()
        if not Path(path).exists():
            return None
        audio_tensor, _ = torchaudio.load(path, backend="soundfile")
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor
