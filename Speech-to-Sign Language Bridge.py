import speech_recognition as sr
import logging
from text_to_gloss import ISLGlossTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    """
    Main loop: Audio -> Text -> Gloss -> Output
    """
    # 1. Initialize the Speech Processor
    speech_engine = SpeechProcessor()
    
    # 2. Initialize your NLP Gloss Translator (from your existing file)
    print("Initializing VISTA NLP Engine...")
    try:
        gloss_engine = ISLGlossTranslator()
    except Exception as e:
        logger.error(f"Failed to load NLP Engine. Ensure 'text_to_gloss.py' is in the same folder. Error: {e}")
        return

    print("--- VISTA Speech-to-Sign Language Pipeline ---")
    print("Press Ctrl+C to stop the program.\n")

    try:
        while True:
            # Step 1: Capture Speech
            english_text = speech_engine.listen_and_recognize()
            
            if english_text:
                print(f"-> Input Text: {english_text}")
                
                # Step 2: Translate to ISL Gloss
                isl_gloss = gloss_engine.translate(english_text)
                
                # Step 3: Output (In a real app, this sends data to the 3D Avatar)
                print(f"-> ISL Gloss: {isl_gloss}")
                print("-" * 40)
            
            # Small interaction loop
            user_choice = input("Press Enter to listen again, or type 'q' to quit: ")
            if user_choice.lower() == 'q':
                break

    except KeyboardInterrupt:
        print("\nStopping VISTA...")

if __name__ == "__main__":
    main()