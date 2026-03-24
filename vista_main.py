
import logging
import speech_recognition as sr
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from advanced_text_to_gloss import AdvancedGlossTranslator
from sign_language_player import SignLanguagePlayer

# ... (Logging config remains) ...

# ... (Logging config remains) ...
# ... (Part 1 imports handled) ...

# --- Part 2: Speech Processor ---

class SpeechProcessor:
    """
    Handles Audio Input and converts it to text using Google Speech Recognition.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust energy threshold for better silence detection
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

    def listen_and_recognize(self) -> str:
        """
        Listens to the microphone and returns recognized text.
        Returns None if recognition fails or times out.
        """
        try:
            with sr.Microphone() as source:
                print("\nListening... (Speak now)")
                
                # Adjust for ambient noise to handle background chatter
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with a timeout to prevent hanging if no audio is detected
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("Processing audio...")
                
                # Use Google Web Speech API (Free tier, no API key required for low volume)
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized Speech: '{text}'")
                return text

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out. No speech detected.")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio.")
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            print("API Error. Please check internet connection.")
            return None
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            return None


# ...

def main():
    print("Initializing VISTA Engine...")
    try:
        gloss_engine = AdvancedGlossTranslator()
        speech_engine = SpeechProcessor()
        # Initialize Player (using absolute path to prevent CWD issues)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(base_dir, "videos")
        sign_player = SignLanguagePlayer(video_dir=video_path) 
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return

    print("\n===========================================")
    print("   VISTA - Virtual Indian Sign Language Translation     ")
    print("===========================================")
    
    while True:
        print("\nSelect Input Mode:")
        print("1. Keypad/Text Input")
        print("2. Voice/Mic Input")
        print("Q. Quit")
        
        choice = input("\nEnter choice (1/2/Q): ").strip().upper()
        
        if choice == '1':
            # Text Mode Loop
            print("\n-- Text Input Mode -- (Type 'back' to return to menu)")
            while True:
                text = input("Enter English Text: ").strip()
                if text.lower() == 'back':
                    break
                if not text:
                    continue
                
                gloss = gloss_engine.translate(text)
                print(f"-> ISL Gloss: {gloss}")
                
                # Play Video
                sign_player.play_sequence(gloss)
        
        elif choice == '2':
            # Voice Mode Loop
            print("\n-- Voice Input Mode -- (Press Ctrl+C to stop listening loop)")
            try:
                while True:
                    text_input = speech_engine.listen_and_recognize()
                    
                    if text_input:
                        print(f"-> Input Text: {text_input}")
                        gloss = gloss_engine.translate(text_input)
                        print(f"-> ISL Gloss: {gloss}")
                        
                        # Play Video
                        sign_player.play_sequence(gloss)
                    
                    # Pause to let user read or choose to continue
                    user_cont = input("\nPress Enter to listen again, or type 'back' to menu: ")
                    if user_cont.lower() == 'back':
                        break
            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                
        elif choice == 'Q':
            print("Exiting VISTA. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
