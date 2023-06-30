import os
from preprocess_utils import AudioPreprocessor, AudioAnalyzer, PromptProcessor


# These times and names were transcribed from the video at https://www.youtube.com/watch?v=lDC4X8Dgxr4
cut_times = [
    '00:43', '01:27', '02:01', '02:33', '02:41', '04:06', '05:33', '06:56',
    '07:57', '09:18', '10:58', '13:28', '15:45', '16:27', '16:34', '17:04', '17:13',
    '19:51', '22:34', '22:45', '23:55', '25:02'
]

names = [
    'Introduction', 'Title Screen', 'Password Screen', 'Stage Select', 'Enemy Chosen',
    'Quick Man', 'Metal Man', 'Bubble Man', 'Heat Man', 'Wood Man', 'Air Man', 'Crash Man',
    'Flash Man', 'Boss Battle', 'Victory', 'Get Your Weapons Ready', 'Dr. Wily\'s Map',
    'Dir Wily\'s Castle', 'Dr. Wily\'s Castle Stage II', 'Dr. Wily Defeated!', 'Epilogue',
    'Credits', 'Game Over'
]

# Initialize audio preprocessor with the audio from the video
audio_file = "data/nes_mega_man_2_soundtrack.wav"
output_dir = "data/megaman/"

audio_preprocessor = AudioPreprocessor(
    filename=audio_file, 
    out_dir=output_dir
)

print('Preprocessing audio...')
sliding_windows = audio_preprocessor.preprocess_audio_file(
    # Cut at cut times
    cut_times=cut_times, 
    # Offset the cut start time by 1 second to omit transition silence
    start_offset=1000, 
    # Offset the cut end time by 3 seconds to omit volume fade
    end_offset=3000,
    # Name each segment according to names
    names=names, 
    # Break each segment into 35 second windows
    window_size=35000,
    # Each window should overlap the previous by 5 seconds
    overlap=5000,
    # Because segments may not fit into 35 second windows, include the remainder in a final window.
    # This introduces additional redundancy to the data, but partially mitigages bias introduced by
    # tending to exclude the ends of segments.
    include_end=True, 
    # artist
    artist='Mega Man 2', 
    # detect cut times
    detect_cut_times=True,
)



# # Get audio features using methods implemented in AudioAnalyzer
# # Currently, just tempo
print('Analyzing audio...')
tsv_filepath = os.path.join(output_dir, 'data_map.tsv')
analyzer = AudioAnalyzer(tsv_filepath)
analyzer.analyze_audio_files()
analyzer.to_tsv(tsv_filepath)

# # Get prompts from the video using methods implemented in PromptProcessor

prompt_processor = PromptProcessor(
    prompt_template = "{artist} {track_name} {tempo_description}",
    tsv_filepath = tsv_filepath,
    out_dir = output_dir
)

prompt_processor.format_prompts()
prompt_processor.to_tsv(tsv_filepath)


# I have a tsv that looks like this: 

# ```
# filename	track_name	artist	tempo	tempo_description
# data/megaman/Introduction_0s_35s.wav	Introduction	megaman	112.0	Moderately fast
# data/megaman/Introduction_4s_39s.wav	Introduction	megaman	112.0	Moderately fast
# data/megaman/Title Screen_0s_35s.wav	Title Screen	megaman	185.0	Very Fast
# data/megaman/Title Screen_5s_40s.wav	Title Screen	megaman	89.0	Moderately slow
# data/megaman/Quick Man_0s_35s.wav	Quick Man	megaman	123.0	Fast
# data/megaman/Quick Man_30s_65s.wav	Quick Man	megaman	123.0	Fast
# ```

# Add a torch Dataset that is implemented like this: 

