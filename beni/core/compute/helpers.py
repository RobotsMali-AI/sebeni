
import numpy as np
from typing import List, Tuple, Union
from beni.core import Token

def levenshtein_matrix(ref: List[str], hyp: List[str]) -> np.ndarray:
    """ Build Levenshtein Matrix """
    m, n = len(ref), len(hyp)

    ld = np.zeros((m+1, n+1), dtype=np.int32)
    
    ld[:, 0] = np.arange(m + 1)
    ld[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            ld[i, j] = min(ld[i-1, j] + 1, ld[i, j-1] + 1, ld[i-1, j-1] + cost)
    
    return ld

def _traceback_ld_ops(ld: np.ndarray, ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    """ Trace Levenshtein Matrix to compute operations count"""
    m, n = len(ref), len(hyp)
    i, j = m, n
    
    s_m, d_m, i_m = 0, 0, 0
    
    while i > 0 or j > 0:

        if(i == 0 or j == 0):
            if(i == 0):
                i_m += 1
            else:
                d_m += 1
            break

        cost = 0 if ref[i-1] == hyp[j-1] else 1

        if(ld[i, j] == ld[i-1, j-1] + cost):
            if cost == 1: 
                s_m += 1
            i -= 1
            j -= 1
        elif(ld[i, j] == ld[i-1, j] + 1):
            d_m += 1
            i -= 1
        else:
            i_m += 1
            j -= 1
    
    return s_m, d_m, i_m


def to_morpheme_list(source: Union[Token, List[str]]) ->List[str]:
    """ Normalize input to list of morpheme strings """
    if isinstance(source, Token):
        return source.best_morphemes()
    return list(source)

def stage_to_phi(stage: Union[int, str], empr: float= 0.5, eps: float= 1e-8) -> float:
    """ Map stage value to phi score. """

    if isinstance(stage, str):
        try:
            stage = int(stage)
        except (ValueError, TypeError):
            return 1.0

    if stage < 0:
        return 0.0
    elif stage < 6:
        return 1.0
    elif stage == 6:
        return empr
    else:
        return eps

def recognized_word(stage: Union[int, str]) -> bool:
    """ Check if a word is recognized by the dictionary """
    
    if isinstance(stage, str):
        try:
            stage = int(stage)
            if (stage >= 0 and stage < 4) or stage == 5:
                return True
            return False
        except (ValueError, TypeError):
            return True

    return (stage >= 0 and stage < 4) or stage == 5
