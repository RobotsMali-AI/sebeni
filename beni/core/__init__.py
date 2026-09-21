"""Core utilities for the Beni package."""
from dataclasses import dataclass, field
from typing import List, Union, Optional

from beni.core.language import Language


@dataclass
class Morpheme:
    """Single morpheme with linguistic annotations."""
    form: str
    ps: List[str] = field(default_factory=list)
    gloss: str = ""
    morphemes: Optional[List["Morpheme"]] = None  # Nested morphemes
    
    def to_list(self) -> List[str]:
        """Flatten morpheme tree into a list of forms."""
        if self.morphemes:
            return [m.form for m in self.morphemes]
        return [self.form]


@dataclass  
class Analysis:
    """One possible morphological analysis of a token."""
    form: str
    ps: List[str] = field(default_factory=list)
    gloss: str = ""
    morphemes: List[Morpheme] = field(default_factory=list)
    
    def morpheme_forms(self) -> List[str]:
        """Extract flat list of morpheme forms."""
        forms = []
        for m in self.morphemes:
            forms.extend(m.to_list())
        return forms if forms else [self.form]


@dataclass
class Token:
    """Annotated token with all its analyses."""
    surface: str
    stage: Union[int, str]
    analyses: List[Analysis] = field(default_factory=list)
    
    def best_morphemes(self, analysis_idx: int = 0) -> List[str]:
        """Get morpheme forms from a specific analysis."""
        if not self.analyses:
            return [self.surface]
        return self.analyses[analysis_idx].morpheme_forms()
    
    @property
    def has_valid_stage(self) -> bool:
        """Check if token has a valid morphological stage."""
        try:
            return int(self.stage) != -1
        except (ValueError, TypeError):
            return True  # tokenizer, g.disamb, etc. are valid


@dataclass
class Sentence:
    """A sentence with its token-level annotations."""
    text: str
    lang: str = "bam"
    language: Optional[Language] = None
    tokens: List[Token] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.language is None and self.lang:
            self.language = Language.from_code(self.lang)
        elif self.language is not None and not self.lang:
            self.lang = self.language.group_code
    
    def stages(self) -> List[Union[int, str]]:
        """Return list of stage values."""
        return [t.stage for t in self.tokens]
    
    def morpheme_sequences(self, analysis_idx: int = 0) -> List[List[str]]:
        """Return morpheme sequences for each token."""
        return [t.best_morphemes(analysis_idx) for t in self.tokens]


def _morpheme_to_dict(morpheme: Morpheme) -> dict:
    data = {
        "form": morpheme.form,
        "ps": list(morpheme.ps or []),
        "gloss": morpheme.gloss,
        "morphemes": [
            _morpheme_to_dict(child) for child in (morpheme.morphemes or [])
        ],
    }
    return data


def sentence_to_completion_json(sentence: Sentence) -> dict:
    """Serialize a DabaX sentence into the policy completion schema."""
    lang = sentence.language.group_code if sentence.language else sentence.lang
    return {
        "text": sentence.text,
        "lang": lang,
        "tokens": [
            {
                "surface": token.surface,
                "stage": token.stage,
                "analyses": [
                    {
                        "form": analysis.form,
                        "ps": list(analysis.ps or []),
                        "gloss": analysis.gloss,
                        "morphemes": [
                            _morpheme_to_dict(m) for m in (analysis.morphemes or [])
                        ],
                    }
                    for analysis in token.analyses
                ],
            }
            for token in sentence.tokens
        ],
    }


__all__ = [
    "Language",
    "Morpheme",
    "Analysis",
    "Token",
    "Sentence",
    "sentence_to_completion_json",
]
