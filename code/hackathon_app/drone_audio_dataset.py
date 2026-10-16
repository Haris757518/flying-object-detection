"""
PyTorch Audio Dataset for Drone Acoustic Detection and Classification.
Handles WAV audio clips in Data/Audio/ for BACKGROUND, DRONE, HELICOPTER.
"""

import os
import glob
import wave
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

AUDIO_CLASSES = ['BACKGROUND', 'DRONE', 'HELICOPTER']
AUDIO_CLASS_TO_ID = {cls: idx for idx, cls in enumerate(AUDIO_CLASSES)}


def load_wav(file_path):
    """Load a WAV file using Python's standard wave library (no external audio backend required)."""
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        elif sampwidth == 1:
            dtype = np.uint8
        else:
            dtype = np.int16

        audio_np = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

        # Normalize to [-1.0, 1.0]
        max_val = float(2 ** (8 * sampwidth - 1))
        audio_np = audio_np / max_val

        return audio_np, framerate


class DroneAudioDataset(Dataset):
    """
    Dataset for drone acoustic signals.
    Returns: waveform tensor (1, N_samples), class_id, metadata dict
    """
    def __init__(self, data_dir="Data/Audio", target_sample_rate=16000, max_duration_sec=5.0, split_df=None):
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.max_samples = int(target_sample_rate * max_duration_sec) if max_duration_sec else None

        self.samples = []
        wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            prefix = fname.split("_")[0].upper()
            if prefix in AUDIO_CLASS_TO_ID:
                self.samples.append({
                    "path": wav_path,
                    "filename": fname,
                    "class_name": prefix,
                    "class_id": AUDIO_CLASS_TO_ID[prefix]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform, sr = load_wav(item["path"])

        # Basic fixed-length clipping / zero-padding
        if self.max_samples is not None:
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            elif len(waveform) < self.max_samples:
                waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        wave_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        label_tensor = torch.tensor(item["class_id"], dtype=torch.long)

        return wave_tensor, label_tensor, item


def export_audio_manifest(data_dir="Data/Audio", output_csv="dataset_audio_metadata.csv"):
    """Generates a comprehensive CSV manifest of all audio samples."""
    records = []
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        cls_name = fname.split("_")[0].upper()
        if cls_name not in AUDIO_CLASS_TO_ID:
            continue
        try:
            with wave.open(wav_path, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                duration = round(n_frames / float(sr), 3)
                channels = wf.getnchannels()
        except Exception:
            sr, duration, channels = None, None, None

        records.append({
            "filename": fname,
            "path": wav_path,
            "class_name": cls_name,
            "class_id": AUDIO_CLASS_TO_ID[cls_name],
            "duration_sec": duration,
            "sample_rate": sr,
            "channels": channels
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved audio metadata to {output_csv} ({len(records)} clips)")
    return df


if __name__ == "__main__":
    export_audio_manifest()
