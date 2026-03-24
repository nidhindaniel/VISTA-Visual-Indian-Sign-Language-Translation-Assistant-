import os
import logging
from typing import List, Optional
try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    pass # Handled in run time if missing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignLanguagePlayer:
    """
    Handles retrieving and stitching ISL video files.
    """
    def __init__(self, video_dir: str = "."):
        self.video_dir = video_dir
        self.available_files = self._index_files()
        
    def _index_files(self):
        """
        Create a case-insensitive map of filename -> real path.
        """
        mapping = {}
        try:
            for f in os.listdir(self.video_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    # Store key as upper case gloss name (without extension)
                    # e.g. "Hello.mp4" -> Key: "HELLO", Value: "Hello.mp4"
                    name_part = os.path.splitext(f)[0].upper()
                    mapping[name_part] = f
            logger.info(f"Indexed {len(mapping)} videos from: {self.video_dir}")
        except FileNotFoundError:
            logger.error(f"Video directory '{self.video_dir}' not found.")
        return mapping

    def _stitch_fast_ffmpeg(self, file_paths: List[str], output_path: str) -> bool:
        """
        Uses FFmpeg 'concat' demuxer for instant stitching (Direct Stream Copy).
        Requires ffmpeg to be installed/available.
        """
        import subprocess
        
        # Create inputs file
        list_file = os.path.join(self.video_dir, "ffmpeg_inputs.txt")
        try:
            with open(list_file, 'w') as f:
                for path in file_paths:
                    # Escape path if needed, usually absolute paths work safely with 'file' keyword
                    clean_path = path.replace('\\', '/')
                    f.write(f"file '{clean_path}'\n")
            
            # Construct FFMPEG command
            # ffmpeg -f concat -safe 0 -i inputs.txt -c copy output.mp4 -y
            # We try to use the ffmpeg binary from moviepy or system
            
            ffmpeg_binary = "ffmpeg"
            try:
                import imageio_ffmpeg
                ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                # Fallback to moviepy config if imageio_ffmpeg is missing (unlikely)
                try:
                    from moviepy.config import get_setting
                    bin_path = get_setting("FFMPEG_BINARY")
                    if bin_path: ffmpeg_binary = bin_path
                except:
                    pass

            cmd = [
                ffmpeg_binary,
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-y",  # Overwrite
                output_path
            ]
            
            # hiding output
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
            
        except Exception as e:
            logger.warning(f"Fast stitch failed: {e}")
            return False
        finally:
            if os.path.exists(list_file):
                try: os.remove(list_file)
                except: pass

    def generate_video(self, gloss_list: List[str], output_filename: str = "temp_output.mp4") -> Optional[str]:
        """
        Stitches videos for the gloss list and saves to output_filename.
        Returns the absolute path of the generated video or None if failed.
        """
        clips_paths = []
        valid_glosses = []
        
        for gloss in gloss_list:
            gloss_upper = gloss.upper()
            video_file = self.available_files.get(gloss_upper)
            if video_file:
                path = os.path.join(self.video_dir, video_file)
                clips_paths.append(path)
                valid_glosses.append(gloss)
            else:
                logger.warning(f"No video found for gloss: '{gloss}'")
        
        if not clips_paths:
            logger.warning("No clips to stitch.")
            return None

        output_path = os.path.join(self.video_dir, output_filename)

        # 1. Try Turbo Mode (FFmpeg Direct Copy)
        print("Attempting Turbo Mode (FFmpeg Direct Copy)...")
        if self._stitch_fast_ffmpeg(clips_paths, output_path):
            print("Turbo Mode Success! Video generated instantly.")
            return output_path
        
        print("Turbo Mode failed (format mismatch?). Falling back to Re-encoding...")

        # 2. Fallback to MoviePy Re-encoding
        try:
             from moviepy import VideoFileClip, concatenate_videoclips
        except ImportError:
             logger.error("MoviePy not installed.")
             return None

        clips = []
        for path in clips_paths:
            try:
                clips.append(VideoFileClip(path))
            except: pass
            
        try:
            final_clip = concatenate_videoclips(clips, method="compose")
            
            try:
                # Try GPU
                final_clip.write_videofile(output_path, codec="h264_nvenc", audio=False, preset="fast", fps=24)
            except Exception:
                # Fallback CPU
                final_clip.write_videofile(output_path, codec="libx264", audio=False, preset="ultrafast", fps=24)
                
            final_clip.close()
            for c in clips: c.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error generation video: {e}")
            return None

    def play_sequence(self, gloss_list: List[str]):
        """
        Generates and plays the video locally.
        """
        print("\nPreparing Video Sequence...")
        path = self.generate_video(gloss_list)
        if path:
            print("Playing Sign Language Video...")
            os.startfile(path)
        else:
            print("Could not generate video sequence.")

if __name__ == "__main__":
    # Test
    player = SignLanguagePlayer(video_dir=".")
    # player.play_sequence(["A", "HELLO"]) # Example usage
