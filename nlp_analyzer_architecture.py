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

        # Get all entity token positions to skip them as main objective candidates
        entity_token_positions = set()
        for ent in doc.ents:
            if ent.label_ in self.config.valid_entity_types:
                for token in ent:
                    entity_token_positions.add(token.i)

        for token in doc:
            if token.pos_ in self.config.content_pos_tags:
                # Skip tokens in compound modifiers
                if token.text.lower() in tokens_in_compounds:
                    continue

                # Skip tokens that are part of entities (they should be attributes, not main objectives)
                if token.i in entity_token_positions:
                    continue

                importance_score = self._calculate_importance_score(token, doc)
                is_format = self._is_format_descriptor(token)
                is_attribute = self.attribute_manager.is_attribute(token.text)

                content_candidates.append(
                    {
                        "text": token.text,
                        "importance": importance_score,
                        "is_format": is_format,
                        "is_attribute": is_attribute,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "ent_type": token.ent_type_,
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

        # Heavily penalize attributes to prevent them from becoming main objectives
        if self.attribute_manager.is_attribute(token.text):
            score -= 10

        # Heavily penalize entities from becoming main objectives - they should stay as attributes
        if token.ent_type_ and self._is_valid_entity(token):
            score -= (
                8  # Strong penalty to keep entities as attributes, not main objectives
            )

        # Subjects and objects get highest priority
        if token.dep_ in self.config.subject_object_deps:
            score += self.config.subject_object_boost + 5

        # Root of dependency tree
        if token.dep_ == "ROOT":
            score += self.config.root_boost

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

    def _assign_objectives(
        self, content_candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[str]]:
        """Assign main objective and sub-objectives from candidates"""
        main_objective = None
        sub_objectives = []

        if not content_candidates:
            return main_objective, sub_objectives

        # Find highest-scoring non-attribute word as main objective (format descriptors are allowed)
        for candidate in content_candidates:
            if candidate["importance"] > 0 and not self.attribute_manager.is_attribute(
                candidate["text"]
            ):
                main_objective = candidate["text"]
                break

        # If no suitable word found, find any non-attribute content word (including format descriptors)
        if not main_objective:
            for candidate in content_candidates:
                if not self.attribute_manager.is_attribute(candidate["text"]):
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
                and not self.attribute_manager.is_attribute(candidate["text"])
            ):

                sub_objectives.append(candidate["text"])
                if len(sub_objectives) >= self.config.max_sub_objectives:
                    break

        return main_objective, sub_objectives


class AttributeManager:
    """Manages all attribute detection strategies"""

    def __init__(self, nlp_model, config: AnalysisConfig):
        self.nlp = nlp_model
        self.config = config
        self.detectors = self._initialize_detectors()

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
        # First, extract all entities (full spans) to avoid duplicates
        processed_entity_tokens = set()

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

        # Then process individual tokens, skipping those already processed as entities
        for token in doc:
            # Skip tokens that are part of multi-word entities
            if token.i in processed_entity_tokens:
                continue

            token_text = token.text
            token_lower = token_text.lower()

            # Skip tokens in compounds or content words
            if token_lower in tokens_in_compounds or token_lower in content_words:
                continue

            # Check attribute detectors
            attributed = False
            for detector in self.detectors:
                is_match, confidence = detector.detect(token)
                if is_match:
                    category = detector.get_category_name()
                    if category in attributes:
                        attributes[category].append(token_text)
                        attributed = True
                        break

            # If not attributed by detectors, use POS-based classification
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
