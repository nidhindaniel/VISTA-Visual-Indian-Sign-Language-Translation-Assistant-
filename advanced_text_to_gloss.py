import spacy
from typing import List, Optional, Set, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# User-Defined Allowed Vocabulary
ALLOWED_VOCAB = {
    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Alphabet
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    # A
    "AFTER", "AGAIN", "AGAINST", "AGE", "ALL", "ALONE", "ALSO", "AND", "ASK", "AT",
    # B
    "BE", "BEAUTIFUL", "BEFORE", "BEST", "BETTER", "BUSY", "BUT", "BYE",
    # C
    "CAN", "CANNOT", "CHANGE", "COLLEGE", "COME", "COMPUTER",
    # D
    "DAY", "DISTANCE", "DO", "DO NOT", "DOES NOT",
    # E
    "EAT", "ENGINEER",
    # F
    "FIGHT", "FINISH", "FROM",
    # G
    "GLITTER", "GO", "GOD", "GOLD", "GOOD", "GREAT",
    # H
    "HAND", "HANDS", "HAPPY", "HELLO", "HELP", "HER", "HERE", "HIS", "HOME", "HOMEPAGE", "HOW",
    # I
    "INVENT", "IT",
    # K
    "KEEP",
    # L
    "LANGUAGE", "LAUGH", "LEARN",
    # M
    "ME", "MORE", "MY",
    # N
    "NAME", "NEXT", "NOT", "NOW",
    # O
    "OF", "ON", "OUR", "OUT",
    # P
    "PRETTY",
    # R
    "RIGHT",
    # S
    "SAD", "SAFE", "SEE", "SELF", "SIGN", "SING", "SO", "SOUND", "STAY", "STUDY",
    # T
    "TALK", "TELEVISION", "THANK", "THANK YOU", "THAT", "THEY", "THIS", "THOSE", "TIME", "TO", "TYPE",
    # U
    "US",
    # W
    "WALK", "WASH", "WAY", "WE", "WELCOME", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHOLE", "WHOSE", "WHY", "WILL", "WITH", "WITHOUT", "WORDS", "WORK", "WORLD", "WRONG",
    # Y
    "YOU", "YOUR", "YOURSELF"
}

class AdvancedGlossTranslator:
    """
    Advanced VISTA Translation Engine.
    Converts Natural English to Linguistically Accurate ISL Glosses.
    Enforces Strict Vocabulary Limit:
    - Digits 0-9 and A-Z are allowed.
    - Unknown words are FINGERSPELLED (broken into chars).
    - Structural markers mapped to allowed words (PAST->BEFORE, FUTURE->WILL, QUESTION->Q).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded Spacy model: {model_name}")
        except OSError:
            logger.error(f"Spacy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            raise

    def translate(self, text: str) -> List[str]:
        if not text:
            return []

        doc = self.nlp(text)
        raw_glosses = []

        for sent in doc.sents:
            sent_gloss = self._process_sentence(sent)
            raw_glosses.extend(sent_gloss)

        # Enforce Vocabulary (Fingerspelling fallback)
        final_glosses = self._enforce_vocab(raw_glosses)
        
        return final_glosses

    def _enforce_vocab(self, glosses: List[str]) -> List[str]:
        """
        Check each gloss against ALLOWED_VOCAB.
        If present, keep it.
        If not, break into characters (A-Z, 0-9).
        """
        filtered = []
        for g in glosses:
            g_upper = g.upper()
            
            # Direct match
            if g_upper in ALLOWED_VOCAB:
                filtered.append(g_upper)
            else:
                # Fingerspell
                # e.g. APPLE -> ['A', 'P', 'P', 'L', 'E']
                # But filter out characters not in vocab (like punctuation not allowed?)
                # User allowed A-Z and 0-9.
                chars = list(g_upper)
                valid_chars = [c for c in chars if c in ALLOWED_VOCAB] # This effectively filters spaces/punctuation
                if valid_chars:
                    filtered.extend(valid_chars)
        return filtered

    def _process_sentence(self, sent) -> List[str]:
        final_gloss = []
        time_words = []
        main_body = []
        wh_words = []
        neg_words = []
        tense_marker = None
        has_wh = False

        # 1. Analyze Tense from root
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root:
            if root.morph.get("Tense") == ["Past"]:
                tense_marker = "BEFORE"
            elif root.morph.get("Tense") == ["Fut"]:
                tense_marker = "WILL"
            for child in root.children:
                if child.dep_ == "aux" and child.lemma_.lower() in ["will", "shall"]:
                    tense_marker = "WILL"

        # Track handled tokens
        handled_indices = set()

        # 2. Extract Time entities to put at the front
        for ent in sent.ents:
            if ent.label_ in ("DATE", "TIME"):
                time_words.extend([self._lemmatize(t) for t in ent])
                handled_indices.update(t.i for t in ent)

        for token in sent:
            if token.i not in handled_indices:
                if token.lemma_.lower() in ["yesterday", "tomorrow", "today", "now", "later", "morning", "evening"]:
                    time_words.append(self._lemmatize(token))
                    handled_indices.add(token.i)

        if tense_marker == "BEFORE" and time_words:
            tense_marker = None

        # 3. Process the entire sequence chronologically to support multiple clauses
        for token in sent:
            if token.i in handled_indices:
                continue

            # WH-words go to the end
            if token.tag_ in ("WDT", "WP", "WP$", "WRB") and token.lemma_.lower() != "that":
                wh_words.append(self._lemmatize(token))
                has_wh = True
                continue

            # Negation
            if token.dep_ == "neg" or token.lemma_.lower() in ("no", "not", "never"):
                neg_words.append("NOT")
                continue

            # Skip punctuation and stop words
            if token.is_punct or token.is_space:
                continue
            if token.lemma_.lower() in ("be", "a", "an", "the", "to", "is", "are", "am", "was", "were", "will", "shall"):
                continue
            if token.lemma_.lower() == "do" and token.dep_ == "aux":
                continue

            # Add to main body
            main_body.append(self._lemmatize(token))

            # If we encountered negation, append it right after the affected word
            if neg_words:
                main_body.extend(neg_words)
                neg_words = []

        # 4. Question marker
        question_marker = "Q" if sent.text.strip().endswith("?") and not has_wh else None

        # Assemble: Time + Subj/Verb/Obj (chronological) + Tense + WH + Q
        final_gloss.extend(time_words)
        final_gloss.extend(main_body)
        if neg_words:  # any remaining negation
            final_gloss.extend(neg_words)
        if tense_marker:
            final_gloss.append(tense_marker)
        final_gloss.extend(wh_words)
        if question_marker:
            final_gloss.append(question_marker)

        return final_gloss

    def _get_noun_phrase(self, token, handled_indices: Set[int]) -> List[str]:
        if token.i in handled_indices:
            return []
        
        children = [c for c in token.children if c.i not in handled_indices]
        
        adjs = []
        nums = []
        others = [] 
        
        for child in children:
            if self._is_skippable(child):
                 continue

            if child.pos_ == "ADJ":
                adjs.append(child)
            elif child.pos_ == "NUM":
                nums.append(child)
            elif child.dep_ == "compound":
                 others.append(child)
            else:
                others.append(child)

        phrase = []
        
        # Compounds
        for o in others:
             phrase.extend(self._get_noun_phrase(o, handled_indices))
        
        # Head
        phrase.append(self._lemmatize(token))
        handled_indices.add(token.i)
        
        # Adjectives
        for a in adjs:
             phrase.append(self._lemmatize(a))
             handled_indices.add(a.i)
        
        # Numbers
        for n in nums:
             phrase.append(self._lemmatize(n))
             handled_indices.add(n.i)
             
        return phrase

    def _is_skippable(self, token) -> bool:
        if token.pos_ in ("PRON", "NUM", "PROPN") or token.lemma_ == "-PRON-":
            return False
            
        if token.lemma_.lower() in ("be", "a", "an", "the"):
            return True
        
        return False
        
    def _lemmatize_chunk(self, chunk) -> str:
        # Note: Chunks (like "Next Week") might return "NEXT WEEK"
        # Since "NEXT WEEK" isn't in vocabulary as a single string, it will be split: N,E,X,T,W,E,E,K
        # If we want to support multi-word allowed phrases (like "THANK YOU"), we need better tokenization or check.
        # "THANK YOU" is in list. "DO NOT" is in list. 
        # But _lemmatize_chunk currently joins with space.
        # _enforce_vocab checks exact string.
        # So "THANK YOU" would match. "NEXT WEEK"? "NEXT" is in list. "WEEK"? Not in list.
        # If I return "NEXT WEEK", _enforce_vocab sees "NEXT WEEK". Not in vocab. -> N,E,X,T, ,W,E,E,K
        # This is a limitation. Ideallly I should return list of strings.
        # "Next" is in list.
        return " ".join([self._lemmatize(t) for t in chunk])

    def _lemmatize(self, token) -> str:
        if isinstance(token, str):
             return token.upper()
        
        lemma = token.lemma_
        if lemma == "-PRON-":
            val = token.text.upper()
        else:
            val = lemma.upper()
            
        return val

if __name__ == "__main__":
    t = AdvancedGlossTranslator()
    
    test_cases = [
        "I walked to the store.", # Store -> not in list -> S T O R E
        "Do you have five red apples?", # Apple -> A P P L E
        "I will not eat today.", 
        "Thank you",
        "Do not cry"
    ]
    
    print("--- Advanced VISTA Engine Test (Restricted Vocab) ---")
    for text in test_cases:
        gloss = t.translate(text)
        print(f"Input: {text}\nGloss: {gloss}\n")
