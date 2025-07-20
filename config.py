from dataclasses import dataclass, field
from typing import List


@dataclass
class NLPConfig:
    """Central configuration for NLP analysis parameters.

    This is identical to the in-script configuration previously embedded in
    run_spacy_lg.py, but stored in its own module so other scripts can import
    and tweak values without touching the main analyzer code.
    """

    # ---------- Similarity thresholds ----------
    similarity_threshold_format: float = 0.45
    similarity_threshold_artistic: float = 0.6
    similarity_threshold_format_root: float = 0.5
    similarity_threshold_size: float = 0.6
    similarity_threshold_material: float = 0.6
    similarity_threshold_shape: float = 0.55

    # ---------- Importance scoring ----------
    importance_threshold_sub_objectives: int = 3
    max_sub_objectives: int = 2

    # Word-frequency threshold: nouns/proper-nouns with rank lower than this
    # value are considered “generic” and receive a penalty.
    generic_word_rank_threshold: int = 10000

    base_importance_score: int = 1
    entity_boost: int = 2
    subject_object_boost: int = 2
    root_boost: int = 1
    adjective_modifier_boost: int = 1
    compound_penalty: int = 1
    noun_chunk_first_boost: int = 3
    noun_chunk_last_boost: int = 1
    generic_penalty: int = 3
    concrete_noun_boost: int = 2
    subject_indicator_boost: int = 2
    format_descriptor_penalty: int = 3

    # ---------- POS tags / dependency labels ----------
    past_participle_tags: List[str] = field(default_factory=lambda: ["VBN", "VBD"])
    content_pos_tags: List[str] = field(default_factory=lambda: ["NOUN", "PROPN"])
    modifier_deps: List[str] = field(
        default_factory=lambda: ["amod", "compound", "nmod"]
    )
    subject_object_deps: List[str] = field(
        default_factory=lambda: ["nsubj", "nsubjpass", "dobj", "pobj"]
    )

    # ---------- Prototype & vocabulary lists ----------
    artistic_terms: List[str] = field(
        default_factory=lambda: [
            "drawn",
            "painted",
            "carved",
            "sculpted",
            "sketched",
            "illustrated",
            "rendered",
            "crafted",
            "designed",
            "created",
            "handmade",
            "traced",
            "etched",
            "printed",
            "digitized",
            "stylized",
            "engraved",
            "embossed",
        ]
    )

    format_concepts: List[str] = field(
        default_factory=lambda: [
            "image",
            "graphic",
            "design",
            "artwork",
            "illustration",
            "logo",
            "icon",
            "symbol",
            "vector",
            "format",
            "file",
            "style",
        ]
    )

    artistic_concepts: List[str] = field(
        default_factory=lambda: [
            "drawn",
            "painted",
            "artistic",
            "created",
            "handmade",
            "sketched",
        ]
    )

    # Rank below which single-token entities are considered too common and thus ignored.
    entity_generic_rank_threshold: int = 8000

    valid_entity_types: List[str] = field(
        default_factory=lambda: [
            "PERSON",
            "GPE",
            "ORG",
            "EVENT",
            "FAC",
            "LOC",
            "PRODUCT",
        ]
    )

    generic_words: List[str] = field(
        default_factory=lambda: [
            "hand",
            "part",
            "piece",
            "thing",
            "item",
            "kit",
            "set",
            "group",
        ]
    )

    # Seed words representing concrete and abstract concepts for centroid similarity
    concrete_prototypes: List[str] = field(
        default_factory=lambda: [
            "apple",
            "car",
            "tree",
            "dog",
            "building",
        ]
    )

    abstract_prototypes: List[str] = field(
        default_factory=lambda: [
            "idea",
            "concept",
            "emotion",
            "freedom",
            "love",
        ]
    )

    # Colour names now resolved dynamically via webcolors and vector similarity.
    colors: List[str] = field(default_factory=list)

    size_prototypes: List[str] = field(
        default_factory=lambda: [
            "small",
            "large",
            "tiny",
            "huge",
            "medium",
            "big",
            "little",
            "enormous",
            "massive",
            "compact",
            "miniature",
            "gigantic",
            "colossal",
            "petite",
        ]
    )

    material_prototypes: List[str] = field(
        default_factory=lambda: [
            "wood",
            "metal",
            "plastic",
            "glass",
            "ceramic",
            "fabric",
            "leather",
            "stone",
            "concrete",
            "paper",
            "rubber",
            "steel",
            "aluminum",
            "copper",
            "bronze",
            "cotton",
            "silk",
            "wool",
            "canvas",
            "vinyl",
        ]
    )

    shape_prototypes: List[str] = field(
        default_factory=lambda: [
            "round",
            "square",
            "circular",
            "rectangular",
            "triangular",
            "oval",
            "curved",
            "straight",
            "angular",
            "spherical",
            "cylindrical",
            "flat",
            "pointed",
            "sharp",
            "hexagonal",
            "octagonal",
            "diamond",
        ]
    )
