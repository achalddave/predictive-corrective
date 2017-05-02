"""Post process charades submission exactly as Gunnar does it."""
import argparse

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_frame_submission')
    parser.add_argument('output_frame_submission')

    args = parser.parse_args()

    NUM_LABELS = 157
    WINDOW = 1
    with open(args.input_frame_submission) as f:
        video_predictions = {}
        for line in f:
            data = line.strip().split(' ')
            video = data[0]
            frame = data[1]
            frame_predictions = np.asarray(map(float, data[2:]))
            if data[0] in video_predictions:
                video_predictions[data[0]] = np.vstack(
                    (video_predictions[data[0]], frame_predictions))
            else:
                video_predictions[data[0]] = frame_predictions

    output = []
    for video, predictions in tqdm(video_predictions.items()):
        for frame, _ in enumerate(predictions):
            start = max(frame - WINDOW, 0)
            end = min(frame + WINDOW, predictions.shape[0])
            predictions[frame, :] = predictions[start:end, :].mean(axis=0)
        selected = np.linspace(0, 74, 25).round().astype(int)
        for i, frame in enumerate(selected):
            output.append('{video} {frame} {predictions}\n'.format(
                video=video,
                frame=i+1,
                predictions=' '.join(map(str, predictions[frame, :]))))

    with open(args.output_frame_submission, 'wb') as f:
        f.writelines(output)
