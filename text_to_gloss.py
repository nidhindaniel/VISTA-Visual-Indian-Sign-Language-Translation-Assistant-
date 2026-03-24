import spacy
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ISLGlossTranslator:
    """
    A class to translate Natural English Text into Indian Sign Language (ISL) Gloss Structure.
    
    This engine follows ISL Grammar constraints:
    1. Word Order: Subject-Object-Verb (SOV).
    2. Time-First Principle: Time adverbs moved to the beginning.
    3. Interrogatives: Wh-words moved to the end.
    4. Negation: Negative markers moved after the verb.
    5. Stop-word Removal: Articles and auxiliaries removed (unless main verb, though copulas often dropped).
    6. Adjective-Noun Swap: Adjectives follow the noun.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the translator with a Spacy model.
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded Spacy model: {model_name}")
        except OSError:
            logger.error(f"Spacy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            raise

    def translate(self, text: str) -> List[str]:
        """
        Translate English text to ISL Glosses.
        """
        if not text:
            return []

        doc = self.nlp(text)
        glosses = []

        for sent in doc.sents:
            sent_gloss = self._process_sentence(sent)
            glosses.extend(sent_gloss)

        logger.info(f"Translated: '{text}' -> {glosses}")
        return glosses

    def _process_sentence(self, sent) -> List[str]:
        """
        Process a single sentence according to ISL rules.
        Refined processing using chunks to handle Adjective-Noun inversion and accurate phrase grouping.
        """
        segments = {
            "time": [],
            "subject": [],
            "object": [], # Includes attributes, direct objects
            "verb": [],
            "neg": [],
            "wh": [],
            "other": [] # indirect objects, etc.
        }
        
        # Track which tokens are handled so we don't double count
        handled_indices = set()

        # 1. Handle Time Entities and Adverbs first
        for ent in sent.ents:
            if ent.label_ in ("DATE", "TIME"):
                segments["time"].append(self._lemmatize_chunk(ent))
                for token in ent:
                    handled_indices.add(token.i)
        
        # 2. Handle Wh-words, Negation, Time Adverbs (not in ents)
        for token in sent:
            if token.i in handled_indices:
                continue
            
            # Time Adverbs
            if token.lemma_ in ["yesterday", "tomorrow", "today", "now", "later", "morning", "evening"]:
                segments["time"].append(self._lemmatize(token))
                handled_indices.add(token.i)
            # Wh-words
            elif token.tag_ in ("WDT", "WP", "WP$", "WRB"):
                # Special check: sometimes "When" is time, but usually acts as Question in "When are you coming?"
                # If it's a question, it goes to WH list.
                segments["wh"].append(self._lemmatize(token))
                handled_indices.add(token.i)
            # Negation
            elif token.dep_ == "neg" or token.lemma_ in ("no", "not", "never"):
                segments["neg"].append(self._lemmatize(token))
                handled_indices.add(token.i)
            elif token.text == "?":
                segments["wh"].append("?") # Keep question mark at end
                handled_indices.add(token.i)

        # 3. Identify Root and assign Subj/Obj
        # We assume one main structure. Complex sentences are flattened.
        for token in sent:
            if token.i in handled_indices:
                continue
            
            # Skip Stopwords
            if token.is_stop and token.pos_ not in ("VERB", "NOUN", "PROPN", "ADJ", "ADV"):
                # Be careful: 'is', 'am' are stops. 'I', 'you' are stops but needed as Subject.
                # PRON should be kept.
                if token.pos_ == "PRON":
                    pass
                # Keep 'up' in 'give up' (part)
                elif token.dep_ == "prt":
                    pass
                else: 
                     # Check if it's MAIN VERB (ROOT) even if stop (e.g. "do" in "I do")
                     if token.dep_ == "ROOT" and token.pos_ == "VERB":
                         pass
                     else:
                         continue
            
            # Skip 'be' verbs (AUX) often
            if token.lemma_ == "be" and token.pos_ == "AUX":
                continue

            # Determine Role
            if token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                # Collect the whole phrase for this subject
                segments["subject"].append(self._get_full_phrase(token, handled_indices))
            elif token.dep_ in ("dobj", "pobj", "attr", "acomp"):
                segments["object"].append(self._get_full_phrase(token, handled_indices))
            elif token.pos_ == "VERB" or token.dep_ == "ROOT":
                 # If it's a verb, add to verb list. 
                 segments["verb"].append(self._lemmatize(token))
                 handled_indices.add(token.i)
            elif token.pos_ == "ADV":
                # General adverbs (quickly) -> usually near verb or object.
                # Put in Object/Other slot for now to be between Subj and Verb.
                segments["object"].append(self._lemmatize(token))
                handled_indices.add(token.i)
            else:
                # Catch-all for nouns/adjectives not caught in chunks?
                # If it's a dangling noun/adj/propn
                 if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                      segments["object"].append(self._lemmatize(token))
                      handled_indices.add(token.i)

        # Assemble: Time + Subject + Object + Verb + Negation + Wh
        # Note: ISL is SOV.
        
        final_sequence = []
        final_sequence.extend(segments["time"])
        final_sequence.extend(segments["subject"])
        final_sequence.extend(segments["object"])
        
        
        final_sequence.extend(segments["verb"])
        final_sequence.extend(segments["neg"])
        final_sequence.extend(segments["wh"])

        # Flatten list of strings/lists
        flat_gloss = []
        for item in final_sequence:
            if isinstance(item, list):
                flat_gloss.extend(item)
            else:
                flat_gloss.append(item)
        
        return [g for g in flat_gloss if g] # Filter empty strings

    def _get_full_phrase(self, token, handled_indices) -> List[str]:
        """
        Get the phrase rooted at 'token', applying Noun-Adj rule.
        Marks tokens as handled.
        """
        if token.i in handled_indices:
            return []
        
        # Get subtree text, but reordered.
        subtree = list(token.subtree)
        
        local_tokens = []
        for t in subtree:
            if t.i in handled_indices:
                continue
            # Skip stop words in phrase usually? Yes.
            if t.is_stop and t.pos_ not in ("NOUN", "ADJ", "VERB", "PRON"):
                continue
            local_tokens.append(t)
            handled_indices.add(t.i)
        
        if not local_tokens:
            return []
        
        # Heuristic for Noun Phrase in ISL: NOUN + ADJ + QUANTIFIER
        
        nouns = [t for t in local_tokens if t.pos_ in ("NOUN", "PROPN", "PRON")]
        adjs = [t for t in local_tokens if t.pos_ == "ADJ"]
        others = [t for t in local_tokens if t not in nouns and t not in adjs]
        
        phrase_gloss = []
        # Add Nouns (Head first?)
        for n in nouns:
            phrase_gloss.append(self._lemmatize(n))
        for a in adjs:
            phrase_gloss.append(self._lemmatize(a))
        for o in others:
            phrase_gloss.append(self._lemmatize(o))
            
        return phrase_gloss

    def _lemmatize_chunk(self, chunk) -> str:
        """
        Process a chunk (like a Time entity) and lemmatize content.
        Time entities often kept together or simplifed.
        "Next week" -> "NEXT WEEK"
        """
        return " ".join([self._lemmatize(t) for t in chunk if not t.is_stop or t.lemma_ in ("week", "month", "year", "next", "last")])

    def _lemmatize(self, token) -> str:
        """
        Convert token to upper case lemma.
        Strings provided directly are just uppered.
        """
        if isinstance(token, str):
            return token.upper()
        
        lemma = token.lemma_
        if lemma == "-PRON-":
            return token.text.upper()
        return lemma.upper()

if __name__ == "__main__":
    translator = ISLGlossTranslator()
    
    print("--- VISTA Text-to-Gloss Engine ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        text = input("\nEnter English Text: ")
        if text.lower() in ('exit', 'quit'):
            print("Exiting...")
            break
            
        gloss = translator.translate(text)
        print(f"ISL Gloss: {gloss}")
