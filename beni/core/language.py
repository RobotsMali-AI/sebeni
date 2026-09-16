"""Language identity for Sebeni (ISO / group code / optional glottocode)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

from beni.utils import config as cfg


def parse_lang_codes(value: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Split CLI/YAML language values (``bam,mku`` or ``["bam", "mku"]``) into codes."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        return [p.strip().lower() for p in parts if p.strip()]
    out: List[str] = []
    for item in value:
        out.extend(parse_lang_codes(str(item)))
    return out


@dataclass
class Language:
    """ISO-639 and Sebeni group identity for an extremely low-resource language.

    Parameters
    ----------
    iso : str
        ISO 639-3 (or 639-1) code supplied by the user or dataset.
    group_code : str
        Sebeni group code used for Distiller/DabaX baselines (e.g. ``mku`` not ``mlq``).
    glottocode : str, optional
        Glottolog identifier when known.
    name : str, optional
        Display name (language or group).
    """

    iso: str
    group_code: str
    glottocode: Optional[str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        return self.group_code or self.iso

    @classmethod
    def from_code(cls, code: str) -> "Language":
        """Build a Language from an ISO or Sebeni group/variant code.

        Maninka metadata uses group_code ``mku`` while packaged baselines live
        under ``baselines/mlq/``; both codes resolve to the same group.
        """
        needle = str(code or "").strip().lower() or "bam"
        meta = cfg.get_language_metadata(needle)
        if meta:
            group = str(meta.get("group_code") or needle).lower()
            iso = needle if len(needle) in (2, 3) else group
            return cls(
                iso=iso,
                group_code=group,
                glottocode=meta.get("glottocode"),
                name=meta.get("language") or meta.get("group_name") or group,
            )
        iso_row = cfg.get_language_iso(needle)
        group = cfg.get_group_code(needle) or needle
        name = iso_row.get("name") if iso_row else needle
        return cls(iso=needle, group_code=group, name=name)

    @classmethod
    def group_codes(cls, codes: Optional[Union[str, Iterable[str]]]) -> List[str]:
        """Unique Sebeni group codes, preserving order."""
        seen = set()
        ordered: List[str] = []
        for code in parse_lang_codes(codes):
            group = cls.from_code(code).group_code
            if group not in seen:
                seen.add(group)
                ordered.append(group)
        return ordered
