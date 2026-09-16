"""Wordfreq: DabaX + Distiller-checkpoint counts (bamfreq-style, not FastText)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from beni.core.language import Language
from beni.core.morphotactic.dabax import DabaX
from beni.utils import config as cfg


@dataclass
class WordfreqReport:
    """Surface / lemma / morpheme / stage histograms."""

    language: str
    checkpoint_id: Optional[str] = None
    surfaces: Dict[str, int] = field(default_factory=dict)
    lemmas: Dict[str, int] = field(default_factory=dict)
    morphemes: Dict[str, int] = field(default_factory=dict)
    stages: Dict[str, int] = field(default_factory=dict)
    n_tokens: int = 0
    n_sentences: int = 0

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "checkpoint_id": self.checkpoint_id,
            "n_tokens": self.n_tokens,
            "n_sentences": self.n_sentences,
            "surfaces": self.surfaces,
            "lemmas": self.lemmas,
            "morphemes": self.morphemes,
            "stages": self.stages,
        }

    def write(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "wordfreq.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        for name, table in (
            ("surfaces.tsv", self.surfaces),
            ("lemmas.tsv", self.lemmas),
            ("morphemes.tsv", self.morphemes),
            ("stages.tsv", self.stages),
        ):
            lines = ["form\tcount"]
            for form, count in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"{form}\t{count}")
            (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _walk_morphemes(morpheme) -> List[str]:
    forms = [morpheme.form] if getattr(morpheme, "form", None) else []
    nested = getattr(morpheme, "morphemes", None) or []
    for child in nested:
        forms.extend(_walk_morphemes(child))
    return forms


def count_texts(
    texts: Iterable[str],
    lang: str,
    gram: Optional[Union[str, Path]] = None,
    ldict: Optional[Union[str, Path]] = None,
    checkpoint_id: Optional[str] = None,
) -> WordfreqReport:
    """Tokenize and analyze ``texts`` with DabaX against the current checkpoint.

    Parameters
    ----------
    texts : iterable of str
        Source sentences.
    lang : str
        ISO or Sebeni group code.
    gram, ldict : path, optional
        Distiller checkpoint files. Defaults to DabaX resolution.
    checkpoint_id : str, optional
        Recorded in the report (e.g. ``baseline_v3``).
    """
    group = Language.from_code(lang).group_code
    dabax = DabaX(group, gram=gram, ldict=ldict, process=True, runtime_dir=cfg.get_workdir().runtime)
    surfaces: Counter = Counter()
    lemmas: Counter = Counter()
    morphemes: Counter = Counter()
    stages: Counter = Counter()
    n_sent = 0
    n_tok = 0

    for text in texts:
        if not text:
            continue
        try:
            sentences = dabax.loader(text)
        except Exception:
            continue
        for sent in sentences:
            n_sent += 1
            for tok in sent.tokens:
                n_tok += 1
                surfaces[tok.surface] += 1
                stages[str(tok.stage)] += 1
                if tok.analyses:
                    lemmas[tok.analyses[0].form] += 1
                    for morph in tok.analyses[0].morphemes or []:
                        for form in _walk_morphemes(morph):
                            morphemes[form] += 1
                else:
                    for form in tok.best_morphemes():
                        morphemes[form] += 1

    return WordfreqReport(
        language=group,
        checkpoint_id=checkpoint_id,
        surfaces=dict(surfaces),
        lemmas=dict(lemmas),
        morphemes=dict(morphemes),
        stages=dict(stages),
        n_tokens=n_tok,
        n_sentences=n_sent,
    )
