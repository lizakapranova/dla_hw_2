import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm
import torchaudio
import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from torch.utils.data import Dataset

def find_video(speaker_id, needed_dir_path):
    for candidate in os.listdir(needed_dir_path):
        if candidate.startswith(speaker_id):
            return needed_dir_path / candidate
    return None

class CustomDirDataset(BaseDataset):
    def __init__(self, dataset_zip_path, part='train', *args, **kwargs):
        self._data_dir = ROOT_PATH / 'data'
        print(ROOT_PATH / dataset_zip_path)
        if not self._data_dir.exists():
            print(self._data_dir)
            self._data_dir.mkdir(exist_ok=True, parents=True)
            print('ok?')
            self.load_dataset(ROOT_PATH / dataset_zip_path) # relative path
            print('ok!')

        index = self._get_or_create_index(part)
        super().__init__(index, *args, **kwargs)

    def load_dataset(self, arch_path):
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "dla_dataset").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        shutil.rmtree(str(self._data_dir / "dla_dataset"))

    def _get_or_create_index(self, part):
        index_path = self._data_dir / f'{part}_index.json'
        if index_path.exists():
            return read_json(str(index_path))
        index = []
        audio_path = self._data_dir / 'audio' / part
        video_path = self._data_dir / 'mouths'
        for mix_audio in tqdm(os.listdir(audio_path / 'mix')):
            s1 = mix_audio[:mix_audio.find('_')]
            s2 = mix_audio[mix_audio.find('_') + 1: mix_audio.find('.')]

            index.extend([
                {
                    'mix_path': str(audio_path / 'mix' / mix_audio),
                    'audio_path': str(audio_path / 's1' / mix_audio),
                    'video_path': str(video_path / f"{s1}.npz")
                },
                {
                    'mix_path': str(audio_path / 'mix' / mix_audio),
                    'audio_path': str(audio_path / 's2' / mix_audio),
                    'video_path': str(video_path / f"{s2}.npz")
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

        mix = self.load_audio(mix_path)
        speaker_audio = self.load_audio(audio_path) if audio_path else None
        video = np.load(video_path)

        instance_data = {'mix_audio': mix, 'speaker_audio': speaker_audio, 'video': video}
        return instance_data


    @staticmethod
    def load_audio(path):
        # assert Path(path).exists()
        if not Path(path).exists():
            return None
        audio_tensor, _ = torchaudio.load(path, backend="soundfile")
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor
