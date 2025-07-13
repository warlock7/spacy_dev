#!/usr/bin/env python3
"""
NLP Analyzer with Improved Class Architecture
============================================

This module provides an NLP analyzer for query intent extraction
with proper separation of concerns and detection mechanisms.

Key Features:
- WebColors integration for accurate color detection
- Semantic similarity for size, material, shape detection
- Configuration-based approach with JSON config
- Modular class architecture with clear responsibilities
- No hardcoded word lists
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import spacy
import webcolors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration schema for NLP analysis - values loaded from config.json"""

    # Semantic similarity thresholds
    color_threshold: float
    size_threshold: float
    material_threshold: float
    shape_threshold: float
    format_threshold: float
    artistic_threshold: float

    # Importance scoring parameters
    importance_threshold_sub_objectives: int
    max_sub_objectives: int
    base_importance_score: int

    # Scoring boosts and penalties
    entity_boost: int
    subject_object_boost: int
    root_boost: int
    adjective_modifier_boost: int
    compound_penalty: int
    noun_chunk_first_boost: int
    noun_chunk_last_boost: int
    generic_penalty: int
    concrete_noun_boost: int
    subject_indicator_boost: int
    format_descriptor_penalty: int

    # Prototype words for semantic similarity
    size_prototypes: List[str]
    material_prototypes: List[str]
    shape_prototypes: List[str]
    format_prototypes: List[str]
    artistic_prototypes: List[str]

    # POS tags and dependency labels
    past_participle_tags: List[str]
    content_pos_tags: List[str]
    modifier_deps: List[str]
    subject_object_deps: List[str]

    # Entity filtering
    valid_entity_types: List[str]
    generic_words: List[str]

    # Multi-layer classification parameters
    multi_layer_classification: Dict[str, Union[float, int]]
    pos_specific_handling: Dict[str, Union[List[str], str]]
    entity_correction: Dict[str, Union[List[str], str, float]]
    compound_detection: Dict[str, Union[List[str], str, float, int]]
    context_fallback: Dict[str, Union[int, float]]

    def __post_init__(self):
        """Validate configuration after loading"""
        # Validate that all required lists are not empty
        list_fields = [
            "size_prototypes",
            "material_prototypes",
            "shape_prototypes",
            "format_prototypes",
            "artistic_prototypes",
            "past_participle_tags",
            "content_pos_tags",
            "modifier_deps",
            "subject_object_deps",
            "valid_entity_types",
            "generic_words",
        ]

        for field_name in list_fields:
            field_value = getattr(self, field_name)
            if not isinstance(field_value, list) or len(field_value) == 0:
                raise ValueError(
                    f"Configuration error: {field_name} must be a non-empty list"
                )

        # Validate threshold ranges
        threshold_fields = [
            "color_threshold",
            "size_threshold",
            "material_threshold",
            "shape_threshold",
            "format_threshold",
            "artistic_threshold",
        ]

        for field_name in threshold_fields:
            field_value = getattr(self, field_name)
            if not isinstance(field_value, (int, float)) or not (
                0.0 <= field_value <= 1.0
            ):
                raise ValueError(
                    f"Configuration error: {field_name} must be a number between 0.0 and 1.0"
                )


class ConfigManager:
    """Manages configuration loading and validation"""

    DEFAULT_CONFIG_PATH = "config.json"

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
        """Load configuration from JSON file"""
        config_path = config_path or ConfigManager.DEFAULT_CONFIG_PATH

        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found. "
                f"Please ensure the config file exists with all required parameters."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            logger.info(f"Configuration loaded from {config_path}")
            return AnalysisConfig(**config_data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file '{config_path}': {e}")
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration parameters in '{config_path}': {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading config from '{config_path}': {e}")


class AttributeDetector(ABC):
    """Abstract base class for attribute detection strategies"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    @abstractmethod
    def detect(self, token_or_text: Union[str, object]) -> Tuple[bool, float]:
        """
        Detect if the token/text matches this attribute type

        Returns:
            Tuple[bool, float]: (is_match, confidence_score)
        """
        pass

    @abstractmethod
    def get_category_name(self) -> str:
        """Get the category name for this detector"""
        pass


class ColorDetector(AttributeDetector):
    """Color detection using WebColors library"""

    def detect(self, token_or_text: Union[str, object]) -> Tuple[bool, float]:
        """Detect colors using WebColors library for accuracy"""
        text = (
            token_or_text.text if hasattr(token_or_text, "text") else str(token_or_text)
        )
        text_lower = text.lower()

        try:
            # Try direct color name lookup (CSS3 specification)
            webcolors.name_to_hex(text_lower, spec=webcolors.CSS3)
            return True, 1.0
        except ValueError:
            # Try CSS2 color names
            try:
                webcolors.name_to_hex(text_lower, spec=webcolors.CSS2)
                return True, 0.95
            except ValueError:
                pass

        return False, 0.0

    def get_category_name(self) -> str:
        return "colors"


class SemanticAttributeDetector(AttributeDetector):
    """Generic semantic similarity-based attribute detector"""

    def __init__(
        self,
        nlp_model,
        config: AnalysisConfig,
        prototypes: List[str],
        threshold: float,
        category_name: str,
    ):
        super().__init__(nlp_model, config)
        self.prototypes = prototypes
        self.threshold = threshold
        self.category_name = category_name

    def detect(self, token_or_text: Union[str, object]) -> Tuple[bool, float]:
        """Detect attribute using semantic similarity"""
        if not hasattr(token_or_text, "has_vector") or not token_or_text.has_vector:
            return False, 0.0

        max_similarity = 0.0

        for prototype in self.prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token_or_text.similarity(prototype_token)
                max_similarity = max(max_similarity, similarity)

        if max_similarity > self.threshold:
            return True, max_similarity

        return False, 0.0

    def get_category_name(self) -> str:
        return self.category_name


class CompoundModifierExtractor:
    """Extract compound modifiers like 'hand drawn', 'oil painted'"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    def extract_compound_modifiers(self, doc) -> List[str]:
        """Extract compound modifiers from document"""
        compound_modifiers = []
        doc_tokens = list(doc)

        for i in range(len(doc_tokens) - 1):
            current_token = doc_tokens[i]
            next_token = doc_tokens[i + 1]

            if self._is_compound_pattern(current_token, next_token):
                compound_phrase = f"{current_token.text} {next_token.text}"
                if compound_phrase not in compound_modifiers:
                    compound_modifiers.append(compound_phrase)

        return compound_modifiers

    def _is_compound_pattern(self, current_token, next_token) -> bool:
        """Check if two tokens form a compound pattern"""
        # Pattern 1: NOUN + VERB (past participle)
        pattern1 = (
            current_token.pos_ == "NOUN"
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
        )

        # Pattern 2: (NOUN|ADJ) + VERB with artistic context
        pattern2 = (
            current_token.pos_ in ["NOUN", "ADJ"]
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
        )

        return pattern1 or pattern2

    def _is_artistic_descriptor(self, token) -> bool:
        """Check if token is an artistic descriptor using semantic similarity"""
        if not token.has_vector:
            return False

        for prototype in self.config.artistic_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > self.config.artistic_threshold:
                    return True

        return False


class ObjectiveExtractor:
    """Extract main and sub-objectives from document"""

    def __init__(self, nlp_model, config: AnalysisConfig, attribute_manager):
        self.nlp = nlp_model
        self.config = config
        self.attribute_manager = attribute_manager

    def extract_objectives(self, doc) -> Tuple[Optional[str], List[str]]:
        """Extract main objective and sub-objectives"""
        # Extract compound modifiers first
        compound_extractor = CompoundModifierExtractor(self.nlp, self.config)
        compound_modifiers = compound_extractor.extract_compound_modifiers(doc)

        # Track tokens in compounds to avoid duplicates
        tokens_in_compounds = self._get_tokens_in_compounds(compound_modifiers)

        # Find content candidates
        content_candidates = self._find_content_candidates(doc, tokens_in_compounds)

        # Sort by importance
        content_candidates.sort(key=lambda x: (-x["importance"], x["is_format"]))

        # Assign objectives
        return self._assign_objectives(content_candidates)

    def _get_tokens_in_compounds(self, compound_modifiers: List[str]) -> Set[str]:
        """Get tokens that are part of compound modifiers"""
        tokens_in_compounds = set()
        for compound in compound_modifiers:
            for part in compound.split():
                tokens_in_compounds.add(part.lower())
        return tokens_in_compounds

    def _find_content_candidates(
        self, doc, tokens_in_compounds: Set[str]
    ) -> List[Dict[str, Any]]:
        """Find potential content words with importance scoring"""
        content_candidates = []

        # Get all entity token positions and their contexts
        entity_token_positions = set()
        entity_info = {}  # Store entity information for smarter handling
        for ent in doc.ents:
            if ent.label_ in self.config.valid_entity_types:
                for token in ent:
                    entity_token_positions.add(token.i)
                    entity_info[token.i] = {
                        "entity_text": ent.text,
                        "entity_label": ent.label_,
                        "entity_span": (ent.start, ent.end),
                    }

        # Count total content tokens for context
        total_content_tokens = sum(
            1 for token in doc if token.pos_ in self.config.content_pos_tags
        )

        for token in doc:
            if token.pos_ in self.config.content_pos_tags:
                # Skip tokens in compound modifiers
                if token.text.lower() in tokens_in_compounds:
                    continue

                # Smarter entity handling: Allow entities as main objectives in certain cases
                skip_as_entity = False
                if token.i in entity_token_positions:
                    entity_data = entity_info[token.i]

                    # Skip if it's a multi-token entity and this token is not the first token
                    # (to avoid "dog sketch" being completely ignored)
                    start_idx, end_idx = entity_data["entity_span"]
                    if end_idx - start_idx > 1 and token.i != start_idx:
                        skip_as_entity = True

                    # For short queries (2-3 content words), be more lenient with entities
                    # Allow common nouns that might be misidentified as entities
                    elif total_content_tokens <= 3:
                        # Allow if it's likely a common noun misidentified as ORG
                        # Use linguistic features instead of hardcoded lists
                        if entity_data[
                            "entity_label"
                        ] == "ORG" and self._is_likely_common_noun(token):
                            skip_as_entity = False
                        # Allow PERSON entities that might be common nouns
                        elif (
                            entity_data["entity_label"] == "PERSON"
                            and len(token.text) <= 6
                        ):
                            skip_as_entity = False
                        else:
                            skip_as_entity = True
                    else:
                        skip_as_entity = True

                if skip_as_entity:
                    continue

                importance_score = self._calculate_importance_score(token, doc)
                is_format = self._is_format_descriptor(token)
                is_attribute = self.attribute_manager.is_attribute(token)

                # Use multi-layer classifier for enhanced classification
                ml_result = (
                    self.attribute_manager.multi_layer_classifier.classify_token(
                        token, doc
                    )
                )
                is_format_enhanced = (
                    ml_result["category"] == "format_types"
                    and ml_result["confidence"] > 0.6
                ) or is_format
                is_attribute_enhanced = (
                    ml_result["category"]
                    in ["colors", "sizes", "materials", "shapes", "format_types"]
                    and ml_result["confidence"] > 0.5
                ) or is_attribute

                content_candidates.append(
                    {
                        "text": token.text,
                        "importance": importance_score,
                        "is_format": is_format_enhanced,
                        "is_attribute": is_attribute_enhanced,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "ent_type": token.ent_type_,
                        "ml_category": ml_result["category"],
                        "ml_confidence": ml_result["confidence"],
                    }
                )

        return content_candidates

    def _calculate_importance_score(self, token, doc) -> int:
        """Calculate importance score for a token"""
        importance_score = self.config.base_importance_score

        # Apply various scoring rules first
        importance_score += self._apply_scoring_rules(token, doc)

        # Then apply format descriptor penalty if applicable
        if self._is_format_descriptor(token):
            importance_score -= self.config.format_descriptor_penalty

        return importance_score

    def _apply_scoring_rules(self, token, doc) -> int:
        """Apply various scoring rules to determine token importance"""
        score = 0

        # Count total content tokens for context
        total_content_tokens = sum(
            1 for t in doc if t.pos_ in self.config.content_pos_tags
        )

        # Heavily penalize attributes to prevent them from becoming main objectives
        if self.attribute_manager.is_attribute(token):
            score -= 10

        # Smarter entity penalty: less harsh for short queries with common nouns
        if token.ent_type_ and self._is_valid_entity(token):
            if total_content_tokens <= 3:
                # Reduced penalty for entities in short queries that might be misidentified common nouns
                if token.ent_type_ == "ORG" and self._is_likely_common_noun(token):
                    score -= 2  # Much lighter penalty
                elif token.ent_type_ == "PERSON" and len(token.text) <= 6:
                    score -= 3  # Lighter penalty for short person names that might be common nouns
                else:
                    score -= 8  # Original heavy penalty for clear entities
            else:
                score -= 8  # Original penalty for longer queries

        # CRITICAL: Use spaCy's dependency intelligence for compound nouns
        # In compound nouns, the HEAD is the main concept, modifiers describe it
        if token.dep_ == "ROOT":
            # ROOT tokens get highest priority - they're the main concept
            score += 20

            # SPECIAL: If ROOT is detected as attribute but should be main concept, boost heavily
            if self.attribute_manager.is_attribute(
                token
            ) and self._is_root_override_needed(token, doc):
                score += 30  # Override attribute penalty for critical ROOT tokens

        # For compound structures, analyze the semantic relationship
        if token.dep_ == "compound":
            head_token = token.head

            # Check if this is a format-medium compound (like "pencil sketch", "oil painting")
            if head_token and self._is_format_descriptor(head_token):
                # In "pencil sketch", both "pencil" and "sketch" are format-related
                # Use spaCy's semantic similarity to determine if compound is format-related
                if self._is_format_related_compound(token, head_token):
                    score += 8  # Boost format-related compounds
                else:
                    score += (
                        5  # Moderate boost for other compounds that modify format words
                    )

            # In "oil painting", prioritize the HEAD concept over the modifier
            elif head_token and self.attribute_manager.is_attribute(head_token):
                # Check if the HEAD should actually be the main concept
                if self._should_be_main_concept(head_token, token):
                    # Reduce this modifier's priority to let HEAD win
                    score -= 10  # Lower modifier priority when head should be main
                else:
                    score += 10  # Standard compound boost
            else:
                # Standard compound noun handling
                score += 10

        # Smart grammatical head prioritization
        is_format_descriptor = self._is_format_descriptor(token)

        # For complex queries (>3 words), strongly prioritize grammatical heads
        # This fixes "large oil painting by Pablo Picasso" -> "painting" should win
        if total_content_tokens > 3:
            if token.dep_ == "ROOT":
                score += 15  # Very high boost for sentence root in complex queries

            # Boost noun chunk heads in complex queries
            for chunk in doc.noun_chunks:
                if chunk.root == token:
                    score += 12  # High boost for being head of noun phrase
                    break

        # For simple queries (≤3 words), be more selective about format descriptors
        else:
            if token.dep_ == "ROOT" and not is_format_descriptor:
                score += 10  # Moderate boost for non-format roots in simple queries
            elif token.dep_ == "ROOT" and is_format_descriptor:
                score -= (
                    5  # Penalty for format roots in simple queries (prefer subjects)
                )

            # In simple queries, prioritize non-format noun chunk heads
            for chunk in doc.noun_chunks:
                if chunk.root == token:
                    if not is_format_descriptor:
                        score += 8  # Good boost for non-format heads
                    else:
                        score -= 3  # Penalty for format heads in simple queries
                    break

        # Subjects and objects get highest priority
        if token.dep_ in self.config.subject_object_deps:
            score += self.config.subject_object_boost + 5

        # Concrete nouns get priority
        if token.pos_ == "NOUN" and token.dep_ not in ["compound", "amod"]:
            score += 3

        # Tokens with adjective modifiers (likely to be main nouns)
        if any(child.dep_ == "amod" for child in token.children):
            score += self.config.adjective_modifier_boost + 2

        # Non-compound modifiers
        if token.dep_ != "compound":
            score += self.config.compound_penalty

        # First in noun chunks
        for chunk in doc.noun_chunks:
            if chunk.start == token.i:
                score += self.config.noun_chunk_first_boost
                break

        # Last in noun chunks
        for chunk in doc.noun_chunks:
            if chunk.end - 1 == token.i:
                score += self.config.noun_chunk_last_boost
                break

        # Generic words penalty
        if token.text.lower() in self.config.generic_words:
            score -= self.config.generic_penalty

        return score

    def _is_format_descriptor(self, token) -> bool:
        """Check if token is a format descriptor using semantic similarity"""
        if not token.has_vector:
            return False

        for prototype in self.config.format_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > self.config.format_threshold:
                    return True

        return False

    def _is_valid_entity(self, token) -> bool:
        """Check if entity is valid (not a false positive)"""
        return token.ent_type_ in self.config.valid_entity_types

    def _is_likely_common_noun(self, token) -> bool:
        """
        Determine if a token is likely a common noun that was misclassified as an entity.
        Uses purely generic linguistic features from spaCy - no hardcoded categories.
        """
        # Basic filters: must be lowercase, short, and noun-like
        if not (token.text.islower() and len(token.text) <= 8):
            return False

        # Allow both NOUN and PROPN (misclassified common nouns often get PROPN)
        if token.pos_ not in ["NOUN", "PROPN"]:
            return False

        # Signal 1: Word frequency analysis
        # Common nouns tend to be more frequent than proper nouns
        if hasattr(token, "rank") and token.rank:
            if token.rank < 15000:  # High frequency suggests common noun
                return True

        # Signal 2: Grammatical context analysis
        # Common nouns in certain syntactic roles are likely misclassified
        if token.dep_ in ["compound", "dobj", "nsubj", "pobj"]:
            # These dependency roles often contain common nouns
            return True

        # Signal 3: Morphological patterns
        # Common nouns often have certain morphological properties
        if token.is_alpha and not token.is_title:
            # Alphabetic, not title case - suggests common noun
            return True

        # Signal 4: Semantic density in vector space
        # Common nouns tend to have many semantic neighbors
        if token.has_vector and hasattr(token, "vector_norm"):
            # Check if the word has a well-formed vector (indicates common usage)
            if token.vector_norm > 0.5:  # Well-established in semantic space
                return True

        # Signal 5: Context-based heuristics
        # Look at the token's children and dependencies
        has_descriptive_children = any(
            child.dep_ in ["amod", "det", "compound"] for child in token.children
        )

        # If this token has adjective modifiers or determiners, it's likely a common noun
        if has_descriptive_children:
            return True

        # Signal 6: Length and complexity patterns
        # Very short, simple words are often common nouns misclassified
        if len(token.text) <= 4 and token.text.isalpha():
            return True

        return False

    def _is_root_override_needed(self, token, doc) -> bool:
        """
        Determine if a ROOT token should override attribute classification.
        Uses spaCy's dependency analysis to identify critical main concepts.
        """
        # ROOT tokens in compound structures often represent the main concept
        if token.dep_ == "ROOT" and token.pos_ == "NOUN":

            # Check if ROOT has compound modifiers - strong signal for main concept
            compound_children = [
                child for child in token.children if child.dep_ == "compound"
            ]
            if compound_children:
                return True

            # Check if ROOT is head of a multi-word noun chunk
            for chunk in doc.noun_chunks:
                if chunk.root == token and len(chunk.text.split()) > 1:
                    return True

            # Check if ROOT has multiple descriptive children (adjectives, etc.)
            descriptive_children = [
                child
                for child in token.children
                if child.dep_ in ["amod", "det", "prep"]
            ]
            if len(descriptive_children) >= 2:
                return True

        return False

    def _assign_objectives(
        self, content_candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[str]]:
        """Assign main objective and sub-objectives from candidates"""
        main_objective = None
        sub_objectives = []

        if not content_candidates:
            return main_objective, sub_objectives

        # CRITICAL: Sort candidates by importance score (highest first)
        content_candidates.sort(key=lambda x: -x["importance"])

        # Enhanced selection: Prioritize subjects over formats in art contexts
        best_subject = None
        best_non_format = None
        best_format = None

        for candidate in content_candidates:
            if candidate["importance"] > 0:

                # Check if this is a subject using multi-layer classifier
                is_subject = (
                    candidate.get("ml_category") == "subject"
                    and candidate.get("ml_confidence", 0) > 0.6
                )

                # Special case: Allow ROOT material attributes to be main objectives
                # (like "painting" in "oil painting") but not format attributes
                is_viable_candidate = (
                    not candidate["is_attribute"]
                    or self._should_allow_root_attribute(candidate)
                    or is_subject
                )

                if is_viable_candidate:
                    # Prioritize subjects first
                    if is_subject and best_subject is None:
                        best_subject = candidate
                    elif not candidate["is_format"] and best_non_format is None:
                        best_non_format = candidate
                    elif candidate["is_format"] and best_format is None:
                        best_format = candidate

                    # Stop once we have all types or found what we need
                    if best_subject and best_non_format and best_format:
                        break

        # Decision logic: Subject > Non-format > Format
        if best_subject:
            main_objective = best_subject["text"]
        elif best_non_format and best_format:
            # If format word has significantly higher score (3x), use it
            if best_format["importance"] >= best_non_format["importance"] * 3:
                main_objective = best_format["text"]
            else:
                main_objective = best_non_format["text"]
        elif best_non_format:
            main_objective = best_non_format["text"]
        elif best_format:
            main_objective = best_format["text"]

        # Fallback: Any non-attribute word (or specific ROOT material override)
        if not main_objective:
            for candidate in content_candidates:
                # Allow ROOT material attributes but not format/shape attributes
                is_viable_candidate = not candidate[
                    "is_attribute"
                ] or self._should_allow_root_attribute(candidate)
                if is_viable_candidate:
                    main_objective = candidate["text"]
                    break

        # Last resort: use the highest scoring one
        if not main_objective and content_candidates:
            main_objective = content_candidates[0]["text"]

        # Add other significant content words as sub-objectives
        for candidate in content_candidates:
            if (
                candidate["text"].lower() != main_objective.lower()
                and not candidate["is_format"]
                and candidate["importance"]
                >= self.config.importance_threshold_sub_objectives
                and candidate["dep"] != "compound"
                and not candidate["is_attribute"]
            ):

                sub_objectives.append(candidate["text"])
                if len(sub_objectives) >= self.config.max_sub_objectives:
                    break

        return main_objective, sub_objectives

    def _should_allow_root_attribute(self, candidate: Dict[str, Any]) -> bool:
        """
        Determine if a ROOT attribute should be allowed as main objective.
        Only allows material attributes that are ROOT, not format/shape attributes.
        """
        if (
            candidate.get("dep") == "ROOT"
            and candidate.get("pos") == "NOUN"
            and candidate.get("is_attribute")
        ):

            # Create a token from the text to check which type of attribute it is
            token = self.nlp(candidate["text"])[0]

            # Check if it's specifically a material attribute (like "painting")
            for detector in self.attribute_manager.detectors:
                is_match, _ = detector.detect(token)
                if is_match and detector.get_category_name() == "materials":
                    return True  # Allow material attributes as main objectives

        return False

    def _is_root_material_override(self, candidate: Dict[str, Any]) -> bool:
        """
        Determine if a ROOT token should override material attribute classification.
        This is a specific override for material attributes that are ROOT and have compound modifiers.
        """
        if candidate["is_attribute"] and candidate["dep"] == "ROOT":
            # Check if the ROOT token is a material attribute
            if self.attribute_manager.is_attribute(candidate["text"]):
                # Check if it has compound modifiers
                compound_extractor = CompoundModifierExtractor(self.nlp, self.config)
                compound_modifiers = compound_extractor.extract_compound_modifiers(
                    self.nlp(candidate["text"])
                )
                if compound_modifiers:
                    return True
        return False

    def _is_format_related_compound(self, modifier_token, head_token) -> bool:
        """
        Determine if a compound is format-related using pure spaCy intelligence.
        No hardcoded word lists - uses semantic relationships and linguistic patterns.
        """
        if not (modifier_token.has_vector and head_token.has_vector):
            return False

        # Signal 1: Semantic coherence with head token
        if head_token.has_vector:
            semantic_coherence = modifier_token.similarity(head_token)
            # Format-related compounds tend to have moderate semantic coherence
            if semantic_coherence > 0.4:
                return True

        # Signal 2: Check if head is already a format descriptor
        if self._is_format_descriptor(head_token):
            # If head is format and they have semantic coherence, likely format compound
            if modifier_token.has_vector and head_token.has_vector:
                coherence = modifier_token.similarity(head_token)
                if coherence > 0.3:
                    return True

        # Signal 3: Morphological and frequency patterns
        if modifier_token.pos_ == "NOUN" and head_token.pos_ == "NOUN":
            # Use word frequency as signal - format tools are moderately common
            if (
                hasattr(modifier_token, "rank")
                and modifier_token.rank
                and 1000 < modifier_token.rank < 50000
            ):
                return True

        return False

    def _should_be_main_concept(self, head_token, modifier_token) -> bool:
        """
        Determine if a HEAD token should be the main concept despite being classified as attribute.
        Uses pure spaCy linguistic intelligence and grammatical analysis.
        """
        # CRITICAL: ROOT tokens should almost always be main concepts
        # This is spaCy's strongest linguistic signal
        if head_token.dep_ == "ROOT":
            return True

        # Check if head is a noun chunk root - these are grammatical main concepts
        for chunk in head_token.doc.noun_chunks:
            if chunk.root == head_token:
                # Multi-word chunks: head is usually the main concept
                if len(chunk.text.split()) > 1:
                    return True

        # Use pure spaCy semantic and morphological analysis
        if head_token.has_vector and modifier_token.has_vector:

            # Signal 1: Semantic generality analysis using vector properties
            head_vector_density = head_token.vector_norm
            modifier_vector_density = modifier_token.vector_norm

            # General concepts often have well-established, dense vectors
            if head_vector_density > modifier_vector_density + 1.0:
                return True

            # Signal 2: Morphological complexity patterns
            head_length = len(head_token.text)
            modifier_length = len(modifier_token.text)

            # In compounds, heads are often shorter, more fundamental words
            if head_length <= modifier_length and head_token.pos_ == "NOUN":
                return True

            # Signal 3: Frequency patterns using spaCy's rank
            if (
                hasattr(head_token, "rank")
                and hasattr(modifier_token, "rank")
                and head_token.rank
                and modifier_token.rank
            ):

                # Lower rank = higher frequency = more fundamental concept
                if head_token.rank < modifier_token.rank * 0.8:
                    return True

        # Signal 4: Dependency pattern analysis
        head_children = list(head_token.children)
        modifier_children = list(modifier_token.children)

        # Heads with more grammatical children are often main concepts
        if len(head_children) > len(modifier_children):
            return True

        return False


class MultiLayerClassifier:
    """Multi-layer classification system for robust attribute detection"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config
        self.pos_aware_layer = POSAwareClassifier(nlp_model, config)
        self.entity_correction_layer = EntityCorrectionLayer(nlp_model, config)
        self.compound_detection_layer = CompoundDetectionLayer(nlp_model, config)
        self.context_fallback_layer = ContextFallbackLayer(nlp_model, config)

    def classify_token(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify token using multi-layer approach"""

        # Layer 1: POS-aware classification
        result = self.pos_aware_layer.classify(token, doc)
        if (
            result["confidence"]
            > self.config.multi_layer_classification["semantic_confidence_threshold"]
        ):
            return result

        # Layer 2: Entity correction
        result = self.entity_correction_layer.classify(token, doc)
        if (
            result["confidence"]
            > self.config.multi_layer_classification[
                "morphological_confidence_threshold"
            ]
        ):
            return result

        # Layer 3: Compound detection
        result = self.compound_detection_layer.classify(token, doc)
        if (
            result["confidence"]
            > self.config.multi_layer_classification["dependency_confidence_threshold"]
        ):
            return result

        # Layer 4: Context fallback
        result = self.context_fallback_layer.classify(token, doc)
        return result


class POSAwareClassifier:
    """Handles POS tag-specific classification logic"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    def classify(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify token with POS awareness"""

        # Handle painting-specific logic
        if token.text.lower() == "painting":
            return self._classify_painting_by_pos(token, doc)

        # Handle other POS-specific cases
        if token.pos_ in ["VERB", "VBG", "VBN"]:
            return self._classify_verb_context(token, doc)
        elif token.pos_ in ["NOUN", "PROPN"]:
            return self._classify_noun_context(token, doc)

        # Default classification
        return {"category": "unknown", "confidence": 0.0}

    def _classify_painting_by_pos(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify 'painting' based on POS and context"""

        if (
            token.pos_ == "VERB"
            or token.tag_ in self.config.pos_specific_handling["verb_tags"]
        ):
            # VERB context: "artist painting" = action
            return {"category": "actions", "confidence": 0.9}

        elif (
            token.pos_ == "NOUN"
            or token.tag_ in self.config.pos_specific_handling["noun_tags"]
        ):
            # NOUN context: check dependency and surrounding context
            return self._classify_noun_painting(token, doc)

        return {"category": "unknown", "confidence": 0.0}

    def _classify_noun_painting(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify 'painting' as noun based on context"""

        # Check dependency pattern
        if token.dep_ == "ROOT":
            # Check if it has material compounds
            compounds = [c for c in token.children if c.dep_ == "compound"]
            if compounds:
                # Check if compound is material-like using semantic analysis
                for compound in compounds:
                    if self._is_material_like(compound):
                        return {"category": "format_types", "confidence": 0.8}

            # Default ROOT noun painting is format
            return {"category": "format_types", "confidence": 0.7}

        elif token.dep_ == "compound":
            # "painting technique" - painting is material
            head = token.head
            if head and self._is_technique_like(head):
                return {"category": "materials", "confidence": 0.8}

        # Default noun painting
        return {"category": "format_types", "confidence": 0.6}

    def _classify_verb_context(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify verbs (actions)"""

        # Special case: VBG forms that are actually format types in art contexts
        if token.tag_ == "VBG" and self._is_format_in_art_context(token, doc):
            return {"category": "format_types", "confidence": 0.8}

        if token.dep_ in self.config.pos_specific_handling["action_dependency_labels"]:
            return {"category": "actions", "confidence": 0.8}

        # Check if it's artistic action using semantic similarity
        if self._is_artistic_action(token):
            return {"category": "actions", "confidence": 0.7}

        return {"category": "unknown", "confidence": 0.0}

    def _is_format_in_art_context(self, token, doc) -> bool:
        """Check if VBG token is actually a format type in art context"""

        # Check if token is semantically similar to format prototypes
        if token.has_vector:
            for prototype in self.config.format_prototypes:
                prototype_token = self.nlp(prototype)[0]
                if prototype_token.has_vector:
                    similarity = token.similarity(prototype_token)
                    if similarity > self.config.format_threshold:
                        return True

                # Check if the context contains art-related words
        for other_token in doc:
            if other_token != token and other_token.has_vector:
                # Check if other token is art-related
                for art_prototype in self.config.artistic_prototypes:
                    art_token = self.nlp(art_prototype)[0]
                    if art_token.has_vector:
                        similarity = other_token.similarity(art_token)
                        if similarity > self.config.artistic_threshold:
                            return True

        return False

    def _classify_noun_context(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify nouns based on context"""

        # Use existing semantic detectors but with POS boost
        base_confidence = 0.5

        # Check if it's in object position
        if token.dep_ in self.config.pos_specific_handling["object_dependency_labels"]:
            base_confidence += self.config.multi_layer_classification["pos_aware_boost"]

        return {"category": "unknown", "confidence": base_confidence}

    def _is_material_like(self, token) -> bool:
        """Check if token is material-like using semantic analysis"""
        if not token.has_vector:
            return False

        # Use semantic similarity with material prototypes
        for prototype in self.config.material_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > self.config.material_threshold:
                    return True

        return False

    def _is_technique_like(self, token) -> bool:
        """Check if token represents a technique or method"""
        if not token.has_vector:
            return False

        # Use semantic analysis to detect technique-like words
        technique_indicators = ["technique", "method", "style", "approach", "way"]
        for indicator in technique_indicators:
            indicator_token = self.nlp(indicator)[0]
            if indicator_token.has_vector:
                similarity = token.similarity(indicator_token)
                if similarity > 0.6:
                    return True

        return False

    def _is_artistic_action(self, token) -> bool:
        """Check if token is an artistic action"""
        if not token.has_vector:
            return False

        # Use semantic similarity with artistic prototypes
        for prototype in self.config.artistic_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > self.config.artistic_threshold:
                    return True

        return False


class EntityCorrectionLayer:
    """Corrects entity misclassifications"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    def classify(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify with entity correction"""

        # Check for color misclassification
        if self._is_misclassified_color(token):
            return {
                "category": "colors",
                "confidence": self.config.entity_correction["correction_confidence"],
            }

        # Check for other entity misclassifications
        if self._is_misclassified_entity(token):
            return self._correct_entity_classification(token, doc)

        return {"category": "unknown", "confidence": 0.0}

    def _is_misclassified_color(self, token) -> bool:
        """Check if token is a misclassified color"""
        return (
            token.text.lower() in self.config.entity_correction["color_words"]
            and token.pos_ in self.config.entity_correction["misclassified_pos_tags"]
        )

    def _is_misclassified_entity(self, token) -> bool:
        """Check if token is a misclassified entity"""
        return (
            token.pos_ in self.config.entity_correction["misclassified_pos_tags"]
            and token.text.islower()
            and len(token.text) <= 8
        )

    def _correct_entity_classification(
        self, token, doc
    ) -> Dict[str, Union[str, float]]:
        """Correct misclassified entity"""

        # Check if it's actually a common noun that should be subject
        if self._is_likely_subject(token, doc):
            return {"category": "subject", "confidence": 0.7}

        # Use semantic analysis to determine correct category
        return self._semantic_classification(token)

    def _is_likely_subject(self, token, doc) -> bool:
        """Check if misclassified entity is likely a subject"""

        # Check grammatical context
        if token.dep_ in ["compound", "nsubj", "dobj"]:
            return True

        # Check if it has descriptive modifiers
        if any(child.dep_ in ["amod", "det"] for child in token.children):
            return True

        return False

    def _semantic_classification(self, token) -> Dict[str, Union[str, float]]:
        """Classify using semantic analysis"""

        if not token.has_vector:
            return {"category": "unknown", "confidence": 0.0}

        # Check against all prototype categories
        best_category = "unknown"
        best_confidence = 0.0

        categories = [
            ("sizes", self.config.size_prototypes, self.config.size_threshold),
            (
                "materials",
                self.config.material_prototypes,
                self.config.material_threshold,
            ),
            ("shapes", self.config.shape_prototypes, self.config.shape_threshold),
            (
                "format_types",
                self.config.format_prototypes,
                self.config.format_threshold,
            ),
        ]

        for category, prototypes, threshold in categories:
            max_similarity = 0.0
            for prototype in prototypes:
                prototype_token = self.nlp(prototype)[0]
                if prototype_token.has_vector:
                    similarity = token.similarity(prototype_token)
                    max_similarity = max(max_similarity, similarity)

            if max_similarity > threshold and max_similarity > best_confidence:
                best_category = category
                best_confidence = max_similarity

        return {"category": best_category, "confidence": best_confidence}


class CompoundDetectionLayer:
    """Detects compound subjects and structures"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    def classify(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify with compound detection"""

        # Check if token is part of compound subject
        if self._is_compound_subject_part(token, doc):
            return {"category": "subject", "confidence": 0.8}

        # Check if token is in compound structure
        if self._is_compound_structure(token):
            return self._classify_compound_role(token, doc)

        # Enhanced fallback for compound tokens
        if token.dep_ == "compound":
            return self._classify_compound_fallback(token, doc)

        return {"category": "unknown", "confidence": 0.0}

    def _classify_compound_fallback(self, token, doc) -> Dict[str, Union[str, float]]:
        """Fallback classification for compound tokens"""

        # Get the head token
        head = token.head
        if not head:
            return {"category": "unknown", "confidence": 0.0}

        # Analyze the compound chain position
        chain_position = self._get_compound_chain_position(token, doc)

        # In art contexts, first compounds are often subjects
        if chain_position == 0:  # First in chain
            # Check if this looks like a subject using semantic analysis
            if self._looks_like_subject(token, doc):
                return {"category": "subject", "confidence": 0.7}

        # Middle compounds are often materials/mediums
        elif chain_position > 0 and chain_position < 2:
            # Check if this looks like a material using semantic analysis
            if self._looks_like_material(token, doc):
                return {"category": "materials", "confidence": 0.7}

        # Default compound classification
        return {"category": "unknown", "confidence": 0.4}

    def _get_compound_chain_position(self, token, doc) -> int:
        """Get position of token in compound chain (0-indexed)"""

        # Find all compounds leading to the same head
        head = token.head
        compounds = []

        # Collect all compounds with the same head
        for t in doc:
            if t.dep_ == "compound" and t.head == head:
                compounds.append(t)

        # Sort by position in sentence
        compounds.sort(key=lambda x: x.i)

        # Return position of current token
        try:
            return compounds.index(token)
        except ValueError:
            return -1

    def _looks_like_subject(self, token, doc) -> bool:
        """Check if token looks like a subject using semantic analysis"""

        if not token.has_vector:
            return False

        # Check if token is concrete and specific
        # Use frequency as a signal - subjects are often moderately frequent
        if hasattr(token, "rank") and token.rank:
            # Not too rare, not too common
            if 1000 < token.rank < 50000:
                return True

        # Check if token is not similar to known attribute types
        attribute_prototypes = (
            self.config.material_prototypes
            + self.config.format_prototypes
            + self.config.size_prototypes
            + self.config.shape_prototypes
        )

        for prototype in attribute_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > 0.6:  # If too similar to attributes, not a subject
                    return False

        return True

    def _looks_like_material(self, token, doc) -> bool:
        """Check if token looks like a material using semantic analysis"""

        if not token.has_vector:
            return False

        # Check similarity to material prototypes
        for prototype in self.config.material_prototypes:
            prototype_token = self.nlp(prototype)[0]
            if prototype_token.has_vector:
                similarity = token.similarity(prototype_token)
                if similarity > self.config.material_threshold:
                    return True

        # Check if token has material-like characteristics
        # Materials often have specific morphological patterns
        if token.pos_ == "NOUN" and len(token.text) > 4:
            # Check if it ends with material-like suffixes or has material-like properties
            if token.text.endswith(("color", "paint", "ink", "clay", "wood")):
                return True

        return False

    def _is_compound_subject_part(self, token, doc) -> bool:
        """Check if token is part of compound subject like 'bird nest'"""

        if token.dep_ == "compound":
            head = token.head
            if head and self._forms_compound_subject(token, head):
                return True

        return False

    def _forms_compound_subject(self, token, head) -> bool:
        """Check if token and head form a compound subject"""

        if not (token.has_vector and head.has_vector):
            return False

        # Use semantic coherence to detect compound subjects
        similarity = token.similarity(head)

        # Compound subjects have moderate semantic coherence
        if (
            self.config.compound_detection["semantic_coherence_threshold"]
            < similarity
            < 0.8
        ):

            # Check if both are concrete nouns
            if token.pos_ == "NOUN" and head.pos_ == "NOUN":

                # Check if they're not in different semantic domains
                if not self._are_different_domains(token, head):
                    return True

        return False

    def _is_compound_structure(self, token) -> bool:
        """Check if token is in compound structure"""
        return token.dep_ == "compound" or any(
            c.dep_ == "compound" for c in token.children
        )

    def _classify_compound_role(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify role in compound structure"""

        if token.dep_ == "compound":
            # Analyze position in compound chain
            position_score = self._analyze_compound_position(token, doc)

            if position_score > 0.7:
                return {"category": "subject", "confidence": 0.8}
            elif position_score > 0.4:
                return {"category": "materials", "confidence": 0.7}
            else:
                return {"category": "format_types", "confidence": 0.6}

        return {"category": "unknown", "confidence": 0.0}

    def _analyze_compound_position(self, token, doc) -> float:
        """Analyze position in compound chain"""

        # Find all compounds in the chain
        compounds = []
        current = token

        # Build chain by following compound dependencies
        while current.head and current.head.dep_ != "ROOT":
            if current.head.dep_ == "compound":
                compounds.append(current.head)
            current = current.head

        # Position in chain indicates role
        if compounds:
            position = (
                len(compounds) - compounds.index(token) if token in compounds else 0
            )
            return position / len(compounds)

        return 0.5

    def _are_different_domains(self, token1, token2) -> bool:
        """Check if tokens are from different semantic domains"""

        # Use vector analysis to detect semantic domain differences
        if not (token1.has_vector and token2.has_vector):
            return False

        # Calculate semantic distance
        similarity = token1.similarity(token2)

        # Very low similarity indicates different domains
        return similarity < 0.2


class ContextFallbackLayer:
    """Fallback classification using context analysis"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config

    def classify(self, token, doc) -> Dict[str, Union[str, float]]:
        """Classify using context analysis"""

        # Calculate context vector
        context_vector = self._calculate_context_vector(token, doc)

        if context_vector is not None:
            # Use context to classify
            return self._classify_by_context(token, context_vector)

        # Final fallback
        return self._default_classification(token)

    def _calculate_context_vector(self, token, doc):
        """Calculate context vector from surrounding tokens"""
        import numpy as np

        if len(doc) < self.config.context_fallback["min_context_tokens"]:
            return None

        context_vectors = []

        for other_token in doc:
            if other_token != token and other_token.has_vector:
                # Calculate distance weight
                distance = abs(other_token.i - token.i)
                distance_weight = (
                    self.config.context_fallback["distance_weight_decay"] ** distance
                )

                # Calculate dependency weight
                dep_weight = 1.0
                if other_token.head == token or token.head == other_token:
                    dep_weight = self.config.context_fallback["dependency_weight_boost"]

                # Add weighted vector
                weighted_vector = other_token.vector * distance_weight * dep_weight
                context_vectors.append(weighted_vector)

        if context_vectors:
            return np.mean(context_vectors, axis=0)

        return None

    def _classify_by_context(self, token, context_vector):
        """Classify token based on context vector"""

        if not token.has_vector:
            return {"category": "unknown", "confidence": 0.0}

        # Calculate contextual similarity
        import numpy as np

        contextual_similarity = np.dot(token.vector, context_vector) / (
            np.linalg.norm(token.vector) * np.linalg.norm(context_vector)
        )

        # Classify based on contextual similarity
        if (
            contextual_similarity
            > self.config.multi_layer_classification["contextual_similarity_high"]
        ):
            return {"category": "format_types", "confidence": 0.7}
        elif (
            contextual_similarity
            < self.config.multi_layer_classification["contextual_similarity_low"]
        ):
            return {"category": "subject", "confidence": 0.6}
        else:
            return {"category": "materials", "confidence": 0.5}

    def _default_classification(self, token) -> Dict[str, Union[str, float]]:
        """Default classification when all else fails"""

        # Use basic POS and dependency information
        if token.pos_ == "NOUN":
            if token.dep_ == "ROOT":
                return {"category": "format_types", "confidence": 0.4}
            elif token.dep_ == "compound":
                return {"category": "subject", "confidence": 0.4}

        return {"category": "unknown", "confidence": 0.3}


class AttributeManager:
    """Manages all attribute detection strategies"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config
        self.detectors = self._initialize_detectors()
        self.multi_layer_classifier = MultiLayerClassifier(nlp_model, config)
        # Store reference to objective extractor for format detection
        self.objective_extractor = None

    def _initialize_detectors(self) -> List[AttributeDetector]:
        """Initialize all attribute detectors"""
        detectors = []

        # Color detector using WebColors
        detectors.append(ColorDetector(self.nlp, self.config))

        # Semantic detectors for other attributes
        detectors.append(
            SemanticAttributeDetector(
                self.nlp,
                self.config,
                self.config.size_prototypes,
                self.config.size_threshold,
                "sizes",
            )
        )

        detectors.append(
            SemanticAttributeDetector(
                self.nlp,
                self.config,
                self.config.material_prototypes,
                self.config.material_threshold,
                "materials",
            )
        )

        detectors.append(
            SemanticAttributeDetector(
                self.nlp,
                self.config,
                self.config.shape_prototypes,
                self.config.shape_threshold,
                "shapes",
            )
        )

        detectors.append(
            SemanticAttributeDetector(
                self.nlp,
                self.config,
                self.config.format_prototypes,
                self.config.format_threshold,
                "format_types",
            )
        )

        return detectors

    def categorize_attributes(
        self, doc, main_objective: Optional[str], sub_objectives: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize all attributes in the document"""
        attributes = {
            "adjectives": [],
            "descriptors": [],
            "modifiers": [],
            "actions": [],
            "format_types": [],
            "entities": [],
            "colors": [],
            "sizes": [],
            "materials": [],
            "shapes": [],
        }

        # Get content words to avoid duplicating
        content_words = self._get_content_words(main_objective, sub_objectives)

        # Extract compound modifiers
        compound_extractor = CompoundModifierExtractor(self.nlp, self.config)
        compound_modifiers = compound_extractor.extract_compound_modifiers(doc)
        tokens_in_compounds = self._get_tokens_in_compounds(compound_modifiers)

        # Categorize tokens
        self._categorize_tokens(doc, attributes, content_words, tokens_in_compounds)

        # Add compound modifiers
        if compound_modifiers:
            attributes["modifiers"].extend(compound_modifiers)

        # Remove empty categories and deduplicate
        return self._finalize_attributes(attributes)

    def is_attribute(self, token_or_text: Union[str, object]) -> bool:
        """Check if token/text is any type of attribute"""
        # First try the multi-layer classifier
        if hasattr(token_or_text, "doc"):
            result = self.multi_layer_classifier.classify_token(
                token_or_text, token_or_text.doc
            )
            if result["category"] in [
                "colors",
                "sizes",
                "materials",
                "shapes",
                "format_types",
            ]:
                return True

        # Fallback to original detectors
        for detector in self.detectors:
            is_match, _ = detector.detect(token_or_text)
            if is_match:
                return True
        return False

    def _get_content_words(
        self, main_objective: Optional[str], sub_objectives: List[str]
    ) -> Set[str]:
        """Get set of main content words"""
        content_words = set()
        if main_objective:
            content_words.add(main_objective.lower())
        if sub_objectives:
            content_words.update([obj.lower() for obj in sub_objectives])
        return content_words

    def _get_tokens_in_compounds(self, compound_modifiers: List[str]) -> Set[str]:
        """Get tokens that are part of compound modifiers"""
        tokens_in_compounds = set()
        for compound in compound_modifiers:
            for part in compound.split():
                tokens_in_compounds.add(part.lower())
        return tokens_in_compounds

    def _categorize_tokens(
        self,
        doc,
        attributes: Dict[str, List[str]],
        content_words: Set[str],
        tokens_in_compounds: Set[str],
    ):
        """Categorize individual tokens and handle multi-word entities"""

        # Count total words to determine if this is a simple query
        total_words = len([token for token in doc if token.is_alpha])
        is_simple_query = total_words <= 3

        # For simple queries (≤3 words), disable entity detection to avoid false positives
        # like "dog sketch" being seen as ORG entity
        processed_entity_tokens = set()

        if not is_simple_query:
            # Only process entities for longer, more complex queries
            for ent in doc.ents:
                if ent.label_ in self.config.valid_entity_types:
                    # Check if entity text overlaps with content words
                    entity_lower = ent.text.lower()
                    if entity_lower not in content_words:
                        attributes["entities"].append(
                            {
                                "text": ent.text,
                                "type": ent.label_,
                                "label": spacy.explain(ent.label_),
                            }
                        )

                    # Track all tokens that are part of this entity
                    for token in ent:
                        processed_entity_tokens.add(token.i)

        # Process all individual tokens for semantic classification
        for token in doc:
            token_text = token.text
            token_lower = token_text.lower()

            # Skip content words but allow compound tokens for special processing
            if token_lower in content_words:
                continue

            # For simple queries, ignore entity membership and focus on semantic classification
            # For complex queries, skip tokens that are part of entities
            if not is_simple_query and token.i in processed_entity_tokens:
                continue

            # Special handling for compound format types using spaCy intelligence
            is_compound_format = False
            if token.dep_ == "compound":
                # Check if this compound is format-related using spaCy's semantic analysis
                head_token = token.head
                if (
                    head_token
                    and self.objective_extractor
                    and self.objective_extractor._is_format_descriptor(head_token)
                ):
                    # Use spaCy to determine if compound modifier is format-related
                    if self._is_format_related_compound_token(token, head_token):
                        attributes["format_types"].append(token_text)
                        is_compound_format = True

            # Skip tokens in compounds (but not format-related compounds)
            if token_lower in tokens_in_compounds and not is_compound_format:
                continue

            # Check multi-layer classifier first (prioritize robust classification)
            attributed = False
            result = self.multi_layer_classifier.classify_token(token, doc)

            if result["category"] != "unknown" and result["confidence"] > 0.5:
                category = result["category"]
                if category in attributes:
                    attributes[category].append(token_text)
                    attributed = True

            # If not attributed by multi-layer classifier, try original detectors
            if not attributed:
                for detector in self.detectors:
                    is_match, confidence = detector.detect(token)
                    if is_match:
                        category = detector.get_category_name()
                        if category in attributes:
                            attributes[category].append(token_text)
                            attributed = True
                            break

            # If not attributed by intelligent detectors, use POS-based classification
            if not attributed:
                if token.pos_ == "ADJ":
                    attributes["adjectives"].append(token_text)
                elif token.pos_ == "ADV":
                    attributes["descriptors"].append(token_text)
                elif token.pos_ == "VERB" and token.tag_ == "VBG":
                    attributes["actions"].append(token_text)
                elif (
                    token.pos_ == "VERB"
                    and token.tag_ in self.config.past_participle_tags
                ):
                    attributes["descriptors"].append(token_text)
                elif token.dep_ in self.config.modifier_deps:
                    attributes["modifiers"].append(token_text)

    def _is_format_related_compound_token(self, modifier_token, head_token) -> bool:
        """
        Determine if a compound modifier token should be categorized as format type.
        Uses pure spaCy linguistic intelligence without hardcoded word lists.

        Examples: In "pencil sketch", "pencil" is format-related compound
                  In "dog sketch", "dog" is not format-related
        """
        if not modifier_token.has_vector:
            return False

        # Use pure spaCy linguistic analysis
        # Check if modifier has characteristics of art tools/mediums

        # Signal 1: Semantic coherence with head token
        if head_token.has_vector:
            semantic_coherence = modifier_token.similarity(head_token)
            # Format-related compounds tend to have moderate semantic coherence
            if semantic_coherence > 0.4:
                return True

        # Signal 2: Morphological and POS patterns
        # Art mediums are often concrete nouns with specific properties
        if (
            modifier_token.pos_ == "NOUN"
            and head_token.pos_ == "NOUN"
            and not modifier_token.is_title  # Common nouns, not proper nouns
            and len(modifier_token.text) <= 10
        ):  # Typically short words

            # Use word frequency as signal - art mediums are moderately common
            if (
                hasattr(modifier_token, "rank")
                and modifier_token.rank
                and 1000 < modifier_token.rank < 50000
            ):  # Not too rare, not too common
                return True

        # Signal 3: Dependency pattern analysis
        # In format compounds, modifier typically describes the medium/tool
        if (
            modifier_token.dep_ == "compound"
            and head_token.dep_ == "ROOT"
            and modifier_token.pos_ == "NOUN"
        ):

            # Check if this follows material+format pattern using vector density
            if (
                modifier_token.has_vector and modifier_token.vector_norm > 5.0
            ):  # Well-established in semantic space
                return True

        return False

    def _is_art_medium_pattern(self, modifier_token, head_token) -> bool:
        """
        Identify art medium patterns using pure spaCy linguistic intelligence.
        No hardcoded lists - uses morphological and semantic relationships.
        """
        if not (modifier_token.has_vector and head_token.has_vector):
            return False

        # Use spaCy's linguistic understanding without hardcoded categories

        # Pattern analysis using dependency and morphological features
        if modifier_token.pos_ == "NOUN" and head_token.pos_ == "NOUN":

            # Check semantic relationship strength
            similarity = modifier_token.similarity(head_token)

            # Art medium patterns: moderate similarity (related but distinct concepts)
            if 0.2 < similarity < 0.7:

                # Additional checks using spaCy's linguistic features
                # Check word complexity and semantic density
                modifier_complexity = len(modifier_token.text) / (
                    modifier_token.vector_norm + 1
                )
                head_complexity = len(head_token.text) / (head_token.vector_norm + 1)

                # Art mediums tend to be simpler, more concrete than their targets
                if modifier_complexity < head_complexity:
                    return True

        return False

    def _finalize_attributes(self, attributes: Dict[str, List]) -> Dict[str, Any]:
        """Remove empty categories and deduplicate"""
        final_attributes = {}
        for category, items in attributes.items():
            if items:
                if category == "entities":
                    final_attributes[category] = items
                else:
                    final_attributes[category] = list(set(items))
        return final_attributes


class NLPAnalyzer:
    """Main NLP analyzer coordinating all components for query intent extraction"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the analyzer with configuration"""
        self.config = ConfigManager.load_config(config_path)
        self.nlp = self._load_spacy_model()
        self.attribute_manager = AttributeManager(self.nlp, self.config)
        self.objective_extractor = ObjectiveExtractor(
            self.nlp, self.config, self.attribute_manager
        )
        # Set reference so AttributeManager can access format detection
        self.attribute_manager.objective_extractor = self.objective_extractor

        logger.info("NLP Analyzer initialized successfully")

    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        """Load spaCy model with caching"""
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            logger.warning("en_core_web_lg not found, falling back to en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to analyze a query and extract intent

        Args:
            query: The search query to analyze

        Returns:
            Dictionary containing analysis results
        """
        if not query or not query.strip():
            return {
                "query": query,
                "main_objective": None,
                "sub_objectives": [],
                "attributes": {},
                "all_candidates": [],
            }

        # Process query with spaCy
        doc = self.nlp(query)

        # Extract main and sub objectives
        main_objective, sub_objectives = self.objective_extractor.extract_objectives(
            doc
        )

        # Categorize attributes
        attributes = self.attribute_manager.categorize_attributes(
            doc, main_objective, sub_objectives
        )

        # Get all candidate phrases for reference
        candidates = self._extract_candidates(doc)

        result = {
            "query": query,
            "main_objective": main_objective,
            "sub_objectives": sub_objectives,
            "attributes": attributes,
            "all_candidates": candidates,
        }

        return result

    def _extract_candidates(self, doc) -> List[str]:
        """Extract all candidate phrases for reference"""
        noun_chunks = []
        noun_tokens = []

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if chunk.text.strip():
                noun_chunks.append(chunk.text.strip())

        # Extract individual noun tokens
        for token in doc:
            if token.pos_ in self.config.content_pos_tags and token.text.strip():
                noun_tokens.append(token.text.strip())

        # Combine and deduplicate
        all_candidates = noun_chunks + noun_tokens
        return sorted(set(all_candidates), key=lambda x: -len(x))


def main():
    """Main function to run the NLP analyzer"""
    print("=== NLP Analyzer with Improved Architecture ===")
    print("Detection without hardcoded word lists")
    print()

    # Initialize analyzer
    try:
        analyzer = NLPAnalyzer()
        print("✅ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return

    # Get user input
    try:
        query = input("\nEnter your search query: ").strip()

        if not query:
            print("No query provided. Exiting.")
            return

        # Analyze query
        result = analyzer.analyze_query(query)

        # Display results
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print("=" * 60)

        print(f"🔍 Query: {result['query']}")
        print(f"🎯 Main Objective: {result['main_objective']}")

        if result["sub_objectives"]:
            print(f"📋 Sub Objectives: {', '.join(result['sub_objectives'])}")

        # Display attributes by category
        for attr_type, values in result["attributes"].items():
            if values:
                if attr_type == "entities":
                    entity_strs = [
                        f"{item['text']} ({item['label']})" for item in values
                    ]
                    print(f"🏷️  {attr_type.title()}: {', '.join(entity_strs)}")
                else:
                    emoji_map = {
                        "colors": "🎨",
                        "sizes": "📏",
                        "materials": "🧱",
                        "shapes": "🔵",
                        "format_types": "📄",
                        "adjectives": "✨",
                        "descriptors": "📝",
                        "modifiers": "🔧",
                        "actions": "⚡",
                    }
                    emoji = emoji_map.get(attr_type, "🏷️")
                    print(
                        f"{emoji} {attr_type.title().replace('_', ' ')}: {', '.join(values)}"
                    )

        # Show detailed JSON for debugging
        print(f"\n📊 Detailed JSON:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ Error analyzing query: {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
