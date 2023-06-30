from pydub import AudioSegment, silence
from typing import List, Dict, Union, Optional
import pandas as pd
import librosa
from tqdm import tqdm

import os
import csv
import numpy as np


class AudioPreprocessor:
    def __init__(self, filename: str, out_dir: str = '.') -> None:
        self.audio: AudioSegment = AudioSegment.from_wav(filename)
        self.out_dir = out_dir

    def write_audio(self, filename: str, audio: AudioSegment) -> None:
        audio.export(filename, format="wav")
    
    def detect_silence(self, min_silence_len: int = 2000, silence_thresh: int = -16) -> List[int]:
        """
        Detect periods of silence in the audio file.

        Args:
            min_silence_len (int): The minimum length of silence to be detected, in milliseconds.
            silence_thresh (int): The maximum volume (in dBFS) that will be considered silent.

        Returns:
            List[int]: A list of cut times (in milliseconds) detected as the end of each silent period.
        """
        silent_periods = silence.detect_silence(self.audio, min_silence_len, silence_thresh)
        cut_times = [end for start, end in silent_periods]
        return cut_times


    def segment_audio(self, cut_times: List[int], names: List[str],
                      start_offset: int = 0, end_offset: int = 0) -> Dict[str, AudioSegment]:
        cut_times = [0] + cut_times + [len(self.audio)]

        segments: Dict[str, AudioSegment] = {}
        for i in range(len(cut_times) - 1):
            start_time = cut_times[i] + start_offset
            end_time = (cut_times[i+1] - end_offset) if cut_times[i+1] > end_offset else cut_times[i+1]
            segment = self.audio[start_time:end_time]
            segments[names[i]] = segment
        return segments

    def sliding_window(self, audio: AudioSegment, window_size: int, overlap: int, include_end: bool = False) -> List[tuple]:
        start = 0
        end = window_size
        windows: List[tuple] = []
        while end <= len(audio):
            windows.append((start, end, audio[start:end]))
            start += (window_size - overlap)
            end += (window_size - overlap)

        # Add a final window that includes the end of the audio segment
        if include_end and start < len(audio) and window_size <= len(audio):
            start = len(audio) - window_size
            end = len(audio)
            windows.append((start, end, audio[start:end]))

        return windows        

    def preprocess_audio_file(
        self, 
        cut_times: Optional[List[Union[int, str]]] = None,
        names: Optional[List[str]] = None, 
        window_size: int = 35000, 
        overlap: int = 5000,
        start_offset: int = 0, 
        end_offset: int = 0,
        include_end: bool = False,
        out_dir: str = None,
        artist: str = None,
        min_silence_len: int = 1000, 
        silence_thresh: int = -16,
        detect_cut_times: bool = False,
    ) -> Dict[str, List[AudioSegment]]:

        if not out_dir:
            out_dir = self.out_dir

        if detect_cut_times and cut_times is None:
            # Detect silence in the audio file to use as cut times
            cut_times = self.detect_silence(min_silence_len, silence_thresh)
            # If names is not provided, name the segments as "Segment 1", "Segment 2", etc.
            names = [f'Segment {i}' for i in range(1, len(cut_times) + 2)]

        elif cut_times is not None and type(cut_times[0]) == str:
            cut_times = [self.convert_timestamp_to_ms(timestamp) for timestamp in cut_times]

        if cut_times is not None:
            segments = self.segment_audio(cut_times, names, start_offset, end_offset)

        else:
            # TODO: this should use the file name as the name or a user specified name
            segments = {"Segment 1": self.audio}

        sliding_windows: Dict[str, List[AudioSegment]] = {}
        
        csv_rows = []
        print('writing audio files')
        for name, segment in segments.items():
            windows = self.sliding_window(segment, window_size, overlap, include_end)
            sliding_windows[name] = windows

            for start, end, window in windows:
                start_sec = start // 1000  # convert to seconds
                end_sec = end // 1000  # convert to seconds
                window_filename = f"{name}_{start_sec}s_{end_sec}s.wav"
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist
                    window_filename = os.path.join(out_dir, window_filename)
                
                # Make audio mono
                window = window.set_channels(1)

                # Resample audio to 32kHz
                window = window.set_frame_rate(32000)
                
                self.write_audio(window_filename, window)

                # Append a row to csv_rows
                csv_rows.append([window_filename, name, artist])

        # Write rows to the CSV file
        csv_filename = os.path.join(out_dir, 'data_map.tsv')
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['filename', 'track_name', 'artist'])  # Write the header
            writer.writerows(csv_rows)  # Write the rows

        return sliding_windows

    @staticmethod
    def convert_timestamp_to_ms(timestamp):
        """
        Convert the given timestamp to milliseconds.

        Args:
            timestamp (str): The timestamp string. Format can be 'hh:mm:ss', 'mm:ss', or 'ss'.

        Returns:
            int: The duration in milliseconds.

        Raises:
            ValueError: If the timestamp format is invalid.

        Examples:
            >>> convert_timestamp_to_ms('01:23:45')
            5025000

            >>> convert_timestamp_to_ms('00:02')
            120000

            >>> convert_timestamp_to_ms('10')
            10000
        """

        parts = timestamp.split(':')
        num_parts = len(parts)

        if num_parts == 3:  # Format: 'hh:mm:ss'
            hours, minutes, seconds = map(int, parts)
        elif num_parts == 2:  # Format: 'mm:ss'
            hours = 0
            minutes, seconds = map(int, parts)
        elif num_parts == 1:  # Format: 'ss'
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            raise ValueError("Invalid timestamp format.")

        total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
        return total_ms


class AudioAnalyzer:
    def __init__(self, tsv_filepath, out_dir: str = '.'):
        self.data = pd.read_csv(tsv_filepath, sep='\t')
        self.out_dir = out_dir

    def get_tempo(self, filename):
        y, sr = librosa.load(filename)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return np.round(tempo)

    def tempo_description(self, tempo):
        if tempo <= 60:
            return "Very slow"
        elif tempo <= 76:
            return "Slow"
        elif tempo <= 108:
            return "Moderately slow"
        elif tempo <= 120:
            return "Moderately fast"
        elif tempo <= 168:
            return "Fast"
        elif tempo <= 200:
            return "Very Fast"
        else:
            return "Extremely Fast"

    # TODO: If this is actually something we want to do, it should be parallelized
    def analyze_audio_files(self):
        tempos = []
        tempo_descriptions = []
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Analyzing audio files"):
            filename = row['filename']
            if os.path.exists(filename):  # Ensure the file exists
                tempo = self.get_tempo(filename)
                tempos.append(tempo)
                tempo_descriptions.append(self.tempo_description(tempo))
            else:
                tempos.append(None)
                tempo_descriptions.append(None)
                print(f"File {filename} not found.")
        
        # Add new columns to the data
        self.data['tempo'] = tempos
        self.data['tempo_description'] = tempo_descriptions

    def to_tsv(self, output_tsv_filepath):
        self.data.to_csv(output_tsv_filepath, sep='\t', index=False)


class PromptProcessor:
    def __init__(self, prompt_template: str, tsv_filepath, out_dir: str = '.'):
        self.data = pd.read_csv(tsv_filepath, sep='\t')
        self.out_dir = out_dir
        self.prompt_template = prompt_template

    def format_prompt(self, row):
        return self.prompt_template.format(**row)
    
    def format_prompts(self, prompt_template: str = None):
        if not prompt_template:
            prompt_template = self.prompt_template
            
        prompts = []
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Writing prompts"):
            prompts.append(self.format_prompt(row))
        
        # Add new columns to the data
        self.data['prompt'] = prompts

    def to_tsv(self, output_tsv_filepath):
        self.data.to_csv(output_tsv_filepath, sep='\t', index=False)
            

