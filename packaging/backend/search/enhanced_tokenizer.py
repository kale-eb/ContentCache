#!/usr/bin/env python3
"""
Enhanced Tokenizer for ContentCache Search
Provides advanced tokenization, spell checking, and word similarity features for improved BM25 search.
"""

import re
import logging
from typing import List, Set, Dict, Optional, Tuple
from pathlib import Path

# Try to import NLTK components with graceful fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try to import spell checker with graceful fallback
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedTokenizer:
    """
    Advanced tokenizer with spell checking, stemming, and intelligent text processing.
    Falls back gracefully to basic regex if advanced features aren't available.
    """
    
    def __init__(self, download_nltk_data: bool = True):
        self.nltk_ready = False
        self.spellchecker_ready = False
        self.stemmer = None
        self.lemmatizer = None
        self.spell_checker = None
        self.stopwords_set = set()
        
        # Common file extensions and technical terms to ignore in spell check
        self.technical_terms = {
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'heic',
            'mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm', 'm4v',
            'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt',
            'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg',
            'iphone', 'android', 'ios', 'macos', 'windows', 'linux',
            'wifi', 'bluetooth', 'usb', 'hdmi', 'cpu', 'gpu', 'ram'
        }
        
        # Common filename patterns
        self.filename_patterns = [
            r'IMG_\d+',           # IMG_1234
            r'DSC\d+',            # DSC1234
            r'\d{4}-\d{2}-\d{2}', # 2024-01-15
            r'\d{8}_\d{6}',       # 20240115_143022
        ]
        
        self._initialize_components(download_nltk_data)
    
    def _initialize_components(self, download_nltk_data: bool):
        """Initialize NLTK and spell checker components."""
        
        # Initialize NLTK
        if NLTK_AVAILABLE:
            try:
                if download_nltk_data:
                    self._download_nltk_data()
                
                # Initialize NLTK components
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                
                # Load stopwords
                try:
                    self.stopwords_set = set(stopwords.words('english'))
                except LookupError:
                    logger.warning("NLTK stopwords not available, using basic set")
                    self.stopwords_set = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
                
                self.nltk_ready = True
                logger.info("✅ NLTK components initialized successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ NLTK initialization failed: {e}")
                self.nltk_ready = False
        else:
            logger.warning("⚠️ NLTK not available, using basic tokenization")
        
        # Initialize spell checker
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker()
                # Add technical terms to spell checker dictionary
                self.spell_checker.word_frequency.load_words(self.technical_terms)
                self.spellchecker_ready = True
                logger.info("✅ Spell checker initialized successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Spell checker initialization failed: {e}")
                self.spellchecker_ready = False
        else:
            logger.warning("⚠️ SpellChecker not available, skipping spell correction")
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else 
                              f'corpora/{package}' if package in ['stopwords', 'wordnet', 'omw-1.4'] else package)
            except LookupError:
                logger.info(f"📥 Downloading NLTK package: {package}")
                try:
                    nltk.download(package, quiet=True)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {package}: {e}")
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return available tokenizer capabilities."""
        return {
            'nltk_tokenization': self.nltk_ready,
            'spell_checking': self.spellchecker_ready,
            'stemming': self.nltk_ready,
            'lemmatization': self.nltk_ready,
            'stopword_filtering': self.nltk_ready or bool(self.stopwords_set)
        }
    
    def tokenize_and_process(self, text: str, 
                           apply_spell_check: bool = True,
                           apply_stemming: bool = True,
                           remove_stopwords: bool = True,
                           min_length: int = 2) -> Tuple[List[str], Dict[str, str]]:
        """
        Advanced tokenization with spell checking, stemming, and filtering.
        
        Returns:
            Tuple of (processed_tokens, corrections_made)
        """
        if not text or not text.strip():
            return [], {}
        
        corrections_made = {}
        
        # Step 1: Initial tokenization
        if self.nltk_ready:
            try:
                raw_tokens = word_tokenize(text.lower())
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}, falling back to regex")
                raw_tokens = self._fallback_tokenize(text)
        else:
            raw_tokens = self._fallback_tokenize(text)
        
        # Step 2: Handle special cases (filenames, technical terms)
        processed_tokens = []
        for token in raw_tokens:
            # Handle filename patterns and technical terms
            if self._is_special_term(token):
                processed_tokens.append(token)
                continue
            
            # Handle camelCase splitting
            if self._is_camel_case(token):
                camel_parts = self._split_camel_case(token)
                processed_tokens.extend(camel_parts)
                continue
            
            processed_tokens.append(token)
        
        # Step 3: Spell checking
        if apply_spell_check and self.spellchecker_ready:
            corrected_tokens = []
            for token in processed_tokens:
                if len(token) >= min_length and token.isalpha() and not self._is_special_term(token):
                    if token not in self.spell_checker:
                        # Find correction
                        correction = self.spell_checker.correction(token)
                        if correction and correction != token:
                            corrections_made[token] = correction
                            corrected_tokens.append(correction)
                        else:
                            corrected_tokens.append(token)
                    else:
                        corrected_tokens.append(token)
                else:
                    corrected_tokens.append(token)
            processed_tokens = corrected_tokens
        
        # Step 4: Stemming/Lemmatization
        if apply_stemming and self.nltk_ready:
            stemmed_tokens = []
            for token in processed_tokens:
                if len(token) >= min_length and token.isalpha() and not self._is_special_term(token):
                    try:
                        # Use lemmatizer for better results, fall back to stemmer
                        if self.lemmatizer:
                            stemmed = self.lemmatizer.lemmatize(token)
                        else:
                            stemmed = self.stemmer.stem(token)
                        stemmed_tokens.append(stemmed)
                    except Exception:
                        stemmed_tokens.append(token)
                else:
                    stemmed_tokens.append(token)
            processed_tokens = stemmed_tokens
        
        # Step 5: Remove stopwords and filter
        if remove_stopwords and self.stopwords_set:
            processed_tokens = [token for token in processed_tokens 
                              if token not in self.stopwords_set and len(token) >= min_length]
        else:
            processed_tokens = [token for token in processed_tokens if len(token) >= min_length]
        
        # Step 6: Remove duplicates while preserving order
        seen = set()
        final_tokens = []
        for token in processed_tokens:
            if token not in seen:
                seen.add(token)
                final_tokens.append(token)
        
        return final_tokens, corrections_made
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        """Basic regex tokenization as fallback."""
        # Enhanced regex that handles contractions better
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        # Handle contractions
        expanded_tokens = []
        for token in tokens:
            if "'" in token:
                # Common contractions
                contractions = {
                    "don't": ["do", "not"],
                    "won't": ["will", "not"],
                    "can't": ["can", "not"],
                    "isn't": ["is", "not"],
                    "aren't": ["are", "not"],
                    "wasn't": ["was", "not"],
                    "weren't": ["were", "not"],
                    "haven't": ["have", "not"],
                    "hasn't": ["has", "not"],
                    "hadn't": ["had", "not"],
                    "wouldn't": ["would", "not"],
                    "shouldn't": ["should", "not"],
                    "couldn't": ["could", "not"],
                    "i'm": ["i", "am"],
                    "you're": ["you", "are"],
                    "we're": ["we", "are"],
                    "they're": ["they", "are"],
                    "it's": ["it", "is"],
                    "he's": ["he", "is"],
                    "she's": ["she", "is"],
                    "i've": ["i", "have"],
                    "you've": ["you", "have"],
                    "we've": ["we", "have"],
                    "they've": ["they", "have"],
                    "i'll": ["i", "will"],
                    "you'll": ["you", "will"],
                    "we'll": ["we", "will"],
                    "they'll": ["they", "will"],
                }
                
                if token in contractions:
                    expanded_tokens.extend(contractions[token])
                else:
                    # Split on apostrophe and keep both parts if meaningful
                    parts = token.split("'")
                    expanded_tokens.extend([p for p in parts if len(p) > 1])
            else:
                expanded_tokens.append(token)
        
        return expanded_tokens
    
    def _is_special_term(self, token: str) -> bool:
        """Check if token is a technical term, file extension, or special pattern."""
        # Check technical terms
        if token.lower() in self.technical_terms:
            return True
        
        # Check filename patterns
        for pattern in self.filename_patterns:
            if re.match(pattern, token, re.IGNORECASE):
                return True
        
        # Check if it looks like a filename or path
        if '.' in token and len(token.split('.')[-1]) <= 4:  # Likely file extension
            return True
        
        return False
    
    def _is_camel_case(self, token: str) -> bool:
        """Check if token is in camelCase or PascalCase."""
        return len(token) > 1 and any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)
    
    def _split_camel_case(self, token: str) -> List[str]:
        """Split camelCase words into separate tokens."""
        # Insert space before uppercase letters
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', token)
        # Split and filter
        parts = [part.lower() for part in spaced.split() if len(part) > 1]
        return parts if parts else [token.lower()]
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for basic use cases."""
        tokens, _ = self.tokenize_and_process(text, 
                                            apply_spell_check=False,
                                            apply_stemming=False, 
                                            remove_stopwords=False)
        return tokens
    
    def get_query_suggestions(self, query: str) -> List[str]:
        """Get spell-corrected suggestions for the query."""
        if not self.spellchecker_ready:
            return [query]
        
        tokens, corrections = self.tokenize_and_process(query, apply_spell_check=True)
        
        if not corrections:
            return [query]
        
        # Build corrected query
        corrected_query = query.lower()
        for original, correction in corrections.items():
            corrected_query = corrected_query.replace(original, correction)
        
        suggestions = [query]
        if corrected_query != query.lower():
            suggestions.append(corrected_query)
        
        return suggestions


# Global tokenizer instance
_tokenizer_instance = None

def get_enhanced_tokenizer() -> EnhancedTokenizer:
    """Get global enhanced tokenizer instance."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = EnhancedTokenizer()
    return _tokenizer_instance 