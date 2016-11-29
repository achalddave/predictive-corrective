import argparse
from moviepy.editor import VideoClip, VideoFileClip, clips_array

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('activations1')
    parser.add_argument('activations2')
    parser.add_argument('activations3')
    parser.add_argument('video')
    parser.add_argument('output')
    parser.add_argument('--frames_per_second', default=10)

    args = parser.parse_args()
    a = VideoFileClip(args.activations1)
    b = VideoFileClip(args.activations2)
    c = VideoFileClip(args.activations3)
    d = VideoFileClip(args.video)
    d = d.resize(a.size)

    final_clip = clips_array([[d, a], [b, c]])
    final_clip.write_videofile(args.output, fps=args.frames_per_second)
