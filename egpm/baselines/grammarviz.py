"""GrammarViz-style SAX baseline (PRD §21, M9).

Do the neural encoder and differentiable HSMM add value over classical,
non-learned grammar induction? This baseline implements the GrammarViz
pipeline's core idea: fixed SAX discretization (per-channel z-normalized
breakpoints) then Sequitur-style grammar induction on the resulting
symbol strings, scoring anomaly by the frequency of novel (unprecedented)
n-grams at test time.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def sax_breakpoints(alphabet_size: int) -> np.ndarray:
    """Gaussian quantile breakpoints for SAX (Lin et al. 2003)."""
    from scipy.stats import norm
    if alphabet_size < 2 or alphabet_size > 10:
        raise ValueError("SAX alphabet supports 2..10 symbols")
    qs = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return norm.ppf(qs)


class SAXDiscretizer:
    """Fixed symbolic discretization: z-normalize per channel, then bin at
    Gaussian breakpoints (per PRD §2.3: SAX's fixed breakpoints are exactly
    the non-learned alternative to VQ)."""

    def __init__(self, word_length: int = 4, alphabet_size: int = 4):
        self.word_length = int(word_length)
        self.alphabet_size = int(alphabet_size)
        self.breakpoints = sax_breakpoints(alphabet_size)

    def transform_window(self, X: np.ndarray, mask: np.ndarray) -> str:
        """One window [W, d] -> one SAX word (channels concatenated).

        Each channel is PAA-reduced to `word_length` segments then binned;
        the per-channel symbol strings are concatenated with channel idents
        to preserve channel identity (as the PRD requires for baselines).
        """
        X = np.asarray(X, dtype=np.float64)
        W, d = X.shape
        parts: List[str] = []
        for c in range(d):
            x = X[:, c]
            m = mask[:, c] > 0
            if m.sum() < 2 or np.nanstd(x[m]) < 1e-12:
                sym = "a" * self.word_length  # constant channel -> lowest bin
                parts.append(f"c{c}:{sym}")
                continue
            z = (x - x[m].mean()) / (x[m].std() + 1e-12)
            # PAA: average within word_length equal segments
            seg = np.array_split(z, self.word_length)
            means = np.array([s.mean() for s in seg])
            bins = np.digitize(means, self.breakpoints)
            sym = "".join(chr(ord("a") + int(b)) for b in bins)
            parts.append(f"c{c}:{sym}")
        return "|".join(parts)

    def transform_run(self, X: np.ndarray, mask: np.ndarray) -> List[str]:
        """Windows [N, W, d] -> list of SAX words (one per window)."""
        X = np.asarray(X)
        mask = np.asarray(mask)
        return [self.transform_window(X[i], mask[i]) for i in range(X.shape[0])]


class SequiturGrammar:
    """Minimal Sequitur-style grammar induction over a token stream.

    Learns digram-repeated rules greedily (Nevill-Manning & Witten 1997):
    repeatedly replace the most frequent adjacent pair (digram) with a new
    non-terminal. Used as the 'classical grammar induction' comparator.
    """

    def __init__(self, max_rules: int = 20):
        self.max_rules = max_rules
        self.rules: Dict[str, List[str]] = {}
        self.expanded_counts: Dict[str, int] = {}

    def induce(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Greedy pair-replacement grammar induction."""
        stream = list(tokens)
        rule_id = 0
        while rule_id < self.max_rules:
            pairs: Dict[Tuple[str, str], int] = {}
            for i in range(len(stream) - 1):
                p = (stream[i], stream[i + 1])
                pairs[p] = pairs.get(p, 0) + 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            nt = f"R{rule_id}"
            self.rules[nt] = list(best)
            # replace all occurrences
            i = 0
            new_stream: List[str] = []
            while i < len(stream):
                if i < len(stream) - 1 and (stream[i], stream[i + 1]) == best:
                    new_stream.append(nt)
                    i += 2
                else:
                    new_stream.append(stream[i])
                    i += 1
            stream = new_stream
            rule_id += 1
        self.expanded_counts = {r: self._count_expanded(r, {}) for r in self.rules}
        return self.rules

    def _count_expanded(self, rule: str, seen: dict) -> int:
        if rule in seen:
            return seen[rule]
        seen[rule] = 1  # guard recursion
        total = 1
        for tok in self.rules[rule]:
            if tok in self.rules:
                total += self._count_expanded(tok, seen) - 1
        seen[rule] = total
        return total


class GrammarVizScorer:
    """Anomaly scorer in the GrammarViz spirit: novelty of test n-grams
    against the training grammar (PRD §21 symbolic baseline family)."""

    def __init__(self, ngram: int = 3):
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        self.ngram = ngram
        self.train_ngrams: set = set()
        self.grammar = SequiturGrammar()

    def fit(self, train_words: List[str]) -> "GrammarVizScorer":
        """Memorize training n-grams and induce the grammar (normal-only)."""
        for i in range(len(train_words) - self.ngram + 1):
            self.train_ngrams.add(tuple(train_words[i : i + self.ngram]))
        self.grammar.induce(train_words)
        return self

    def score_run(self, words: List[str]) -> np.ndarray:
        """Per-window novelty score: fraction of the window's n-grams never
        seen in training (0 for windows early enough to have none)."""
        T = len(words)
        out = np.zeros(T)
        for t in range(T):
            lo = max(0, t - self.ngram + 1)
            grams = [tuple(words[i : i + self.ngram])
                     for i in range(lo, t + 1)
                     if i + self.ngram <= T]
            if not grams:
                out[t] = 0.0
                continue
            novel = sum(1 for g in grams if g not in self.train_ngrams)
            out[t] = novel / len(grams)
        return out
