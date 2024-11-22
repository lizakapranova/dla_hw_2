import hydra
from hydra.utils import instantiate
from tqdm.auto import tqdm
import os

from src.utils.io_utils import ROOT_PATH
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.metrics.tracker import MetricTracker


@hydra.main(version_base=None, config_path="src/configs", config_name="calculation")
def main(config):
    gt_path = ROOT_PATH / config.ground_truth
    pred_path = ROOT_PATH / config.predicted

    assert (gt_path / "s1").exists(), (
        "s1 subfolder should exists in ground-truth"
    )
    assert (gt_path / "s2").exists(), (
        "s2 subfolder should exists in ground-truth"
    )
    assert (gt_path / "mix").exists(), (
        "mix subfolder should exists in ground-truth"
    )
    assert (pred_path / "s1").exists(), (
        "s1 subfolder should exists in predicted"
    )
    assert (pred_path / "s2").exists(), (
        "s2 subfolder should exists in predicted"
    )

    metrics = instantiate(config.metrics)

    evaluation_metrics = MetricTracker(
        *[m.name for m in metrics["inference"]],
        writer=None,
    )

    for speaker_id in ['s1', 's2']:
        for gt_audio_path in tqdm(os.listdir(gt_path / speaker_id)):
            gt_audio = CustomDirDataset.load_audio(gt_path / speaker_id / gt_audio_path)
            mix_audio = CustomDirDataset.load_audio(gt_path / 'mix' / gt_audio_path)
            pred_audio = CustomDirDataset.load_audio(pred_path / speaker_id / gt_audio_path)

            assert pred_audio is not None, (
                "prediction missed"
            )

            batch = {"speaker_audio": gt_audio, "mix_audio": mix_audio, "predicted_audio": pred_audio}
            for met in metrics["inference"]:
                evaluation_metrics.update(met.name, met(**batch))

    for key, value in evaluation_metrics.result().items():
            full_key = key
            print(f"{full_key:15s}: {value}")


if __name__ == "__main__":
    main()
    