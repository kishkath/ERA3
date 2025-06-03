from typing import List, Dict, Any
import re
from dataclasses import dataclass
import json

@dataclass
class HeuristicResult:
    modified_query: str
    modified_result: str
    applied_heuristics: List[str]
    confidence_score: float

class QueryHeuristics:
    def __init__(self):
        # Load banned words from config
        self.banned_words = self._load_banned_words()
        self.max_query_length = 500
        self.min_query_length = 3
        
    def _load_banned_words(self) -> List[str]:
        try:
            with open("config/banned_words.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return ["inappropriate", "offensive", "sensitive"]  # Default list
    
    def apply_heuristics(self, query: str, result: str) -> HeuristicResult:
        applied_heuristics = []
        modified_query = query
        modified_result = result
        confidence_score = 1.0
        
        # 1. Remove banned words
        if self._contains_banned_words(query):
            modified_query = self._remove_banned_words(query)
            applied_heuristics.append("banned_words_removal")
            confidence_score *= 0.9
        
        # 2. Normalize whitespace
        if self._has_irregular_whitespace(query):
            modified_query = self._normalize_whitespace(query)
            applied_heuristics.append("whitespace_normalization")
        
        # 3. Check query length
        if len(query) > self.max_query_length:
            modified_query = query[:self.max_query_length]
            applied_heuristics.append("query_length_truncation")
            confidence_score *= 0.8
        
        # 4. Remove special characters
        if self._has_special_characters(query):
            modified_query = self._remove_special_characters(query)
            applied_heuristics.append("special_characters_removal")
        
        # 5. Normalize case
        if self._has_mixed_case(query):
            modified_query = query.lower()
            applied_heuristics.append("case_normalization")
        
        # 6. Remove redundant punctuation
        if self._has_redundant_punctuation(query):
            modified_query = self._remove_redundant_punctuation(query)
            applied_heuristics.append("punctuation_normalization")
        
        # 7. Check for minimum length
        if len(query) < self.min_query_length:
            confidence_score *= 0.5
            applied_heuristics.append("minimum_length_warning")
        
        # 8. Remove duplicate words
        if self._has_duplicate_words(query):
            modified_query = self._remove_duplicate_words(query)
            applied_heuristics.append("duplicate_words_removal")
        
        # 9. Normalize numbers
        if self._has_number_variations(query):
            modified_query = self._normalize_numbers(query)
            applied_heuristics.append("number_normalization")
        
        # 10. Check result relevance
        if not self._is_result_relevant(query, result):
            confidence_score *= 0.7
            applied_heuristics.append("low_relevance_warning")
        
        return HeuristicResult(
            modified_query=modified_query,
            modified_result=modified_result,
            applied_heuristics=applied_heuristics,
            confidence_score=confidence_score
        )
    
    def _contains_banned_words(self, text: str) -> bool:
        return any(word in text.lower() for word in self.banned_words)
    
    def _remove_banned_words(self, text: str) -> str:
        for word in self.banned_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def _has_irregular_whitespace(self, text: str) -> bool:
        return bool(re.search(r'\s{2,}', text))
    
    def _normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def _has_special_characters(self, text: str) -> bool:
        return bool(re.search(r'[^\w\s.,?!-]', text))
    
    def _remove_special_characters(self, text: str) -> str:
        return re.sub(r'[^\w\s.,?!-]', '', text)
    
    def _has_mixed_case(self, text: str) -> bool:
        return bool(re.search(r'[A-Z]', text)) and bool(re.search(r'[a-z]', text))
    
    def _has_redundant_punctuation(self, text: str) -> bool:
        return bool(re.search(r'[.,?!]{2,}', text))
    
    def _remove_redundant_punctuation(self, text: str) -> str:
        return re.sub(r'([.,?!])\1+', r'\1', text)
    
    def _has_duplicate_words(self, text: str) -> bool:
        words = text.split()
        return len(words) != len(set(words))
    
    def _remove_duplicate_words(self, text: str) -> str:
        words = text.split()
        return ' '.join(dict.fromkeys(words))
    
    def _has_number_variations(self, text: str) -> bool:
        return bool(re.search(r'\d+', text))
    
    def _normalize_numbers(self, text: str) -> str:
        return re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', text)
    
    def _is_result_relevant(self, query: str, result: str) -> bool:
        # Simple relevance check based on common words
        query_words = set(query.lower().split())
        result_words = set(result.lower().split())
        common_words = query_words.intersection(result_words)
        return len(common_words) / len(query_words) > 0.3 if query_words else False 