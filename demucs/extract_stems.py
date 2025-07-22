import musdb
import stempeg
import os
subsets = "test" # Replace train or test at subsets and
mus = musdb.DB(root="data/musdb18", is_wav=False, subsets=subsets)  

for track in mus.load_mus_tracks():
    print(f"Processing: {track.name}")
    target_dir = os.path.join("data/musdb18",subsets, track.name)
    os.makedirs(target_dir, exist_ok=True)

    # Save mixture
    stempeg.write_audio(
        os.path.join(target_dir, "mixture.wav"),
        track.audio,
        track.rate
    )

    # Save sources (vocals, drums, etc.)
    for inst in track.sources:
        audio = track.sources[inst].audio
        stempeg.write_audio(
            os.path.join(target_dir, f"{inst}.wav"),
            audio,
            track.rate
        )