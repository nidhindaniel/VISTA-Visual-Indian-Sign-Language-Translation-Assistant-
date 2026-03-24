try:
    from moviepy import VideoFileClip
    print("Direct import worked")
except ImportError:
    print("Direct import failed")

import moviepy
print(f"MoviePy version: {moviepy.__version__}")
# print(dir(moviepy))
