"""
ChapatiLM MV Final: Clean Math Vision Pipeline - STABILIZED VERSION
===================================================================
FIXES:
1. Analytical backprop (no finite-diff)
2. Gradient clipping + feature normalization in arith solver (fixes NaN)
3. Balanced detector training data (fixes 0.881 for everything)
4. Flexible type_map with auto-detection of dataset categories (fixes Unknown)
5. Correct GELU derivative throughout
6. RESOLVED "Unknown" ghosting by removing the epoch cap in type classification.
7. INTEGRATED "STABILIZED PRECISION SCHEDULE" support.
"""

import sys
import os
import re
import json
import math
import random
import time
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import cpuwarp_ml


# ============================================================
# TekkenTokenizer: BPE + R2L digit tokenization
# ============================================================
class TekkenTokenizer:
    def __init__(self, vocab_size: int = 130000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
            "<sep>": 4, "<cls>": 5, "<mask>": 6,
            "<audio>": 7, "<control>": 8, "<tool>": 9,
            "<image>": 10, "<video>": 11,
            "<system>": 12, "<user>": 13, "<assistant>": 14,
        }
        self.vocab = self._build_vocabulary()
        self.merges = self._build_merges()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_lookup = {merge: idx for idx, merge in enumerate(self.merges)}
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s a-zA-Z0-9]+|\s+"
        )
        print(f"Tekken Tokenizer: {len(self.vocab)} tokens, {len(self.merges)} merges")

    def _build_vocabulary(self) -> Dict[str, int]:
        vocab = dict(self.special_tokens)
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        extended = [
            "\u20ac", "\u00a3", "\u00a5", "\u00a9", "\u00ae", "\u2122", "\u00b0", "\u00b1",
            "\u00b5", "\u00b7", "\u00a7", "\u00b6", "\u2020", "\u2021", "\u2022", "\u2026",
            "\u2013", "\u2014", "\u2764", "\ud83d\udd25", "\ud83d\ude80", "\ud83d\udca1",
            "\ud83d\udcca", "\ud83d\udd27", "\ud83d\udcbb", "\ud83d\udcf1", "\ud83c\udf0d",
            "\ud83d\udd12", "\ud83d\udd11", "\ud83d\udcc8", "\ud83d\udcc9", "\ud83d\udcb0",
        ]
        for c in extended:
            if c not in vocab:
                vocab[c] = len(vocab)
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what",
        ]
        for w in common_words:
            if w not in vocab:
                vocab[w] = len(vocab)
        subwords = [
            "ing", "ed", "s", "es", "ly", "tion", "ment", "ness", "ful", "less",
            "un", "re", "pre", "dis", "able", "ible", "al", "ive", "ize", "ate",
            "ify", "hood", "ship", "dom",
        ]
        for sw in subwords:
            if sw not in vocab:
                vocab[sw] = len(vocab)
        byte_pairs = [
            "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es",
            "or", "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "ha", "ng",
            "se", "ou", "io", "le", "ve", "co", "me", "de", "hi", "ri", "ro", "ic",
            "ne", "ea", "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur",
            "ad", "id",
        ]
        for bp in byte_pairs:
            if bp not in vocab:
                vocab[bp] = len(vocab)
        import string
        for c1 in string.ascii_lowercase:
            for c2 in string.ascii_lowercase:
                if len(vocab) >= self.vocab_size:
                    break
                pair = c1 + c2
                if pair not in vocab:
                    vocab[pair] = len(vocab)
            if len(vocab) >= self.vocab_size:
                break
        return vocab

    def _build_merges(self) -> List[Tuple[str, str]]:
        return [
            ("t", "h"), ("h", "e"), ("e", " "), (" ", "t"), ("t", "o"), ("o", " "),
            (" ", "a"), ("a", "n"), ("n", "d"), ("d", " "), (" ", "i"), ("i", "n"),
            ("n", " "), (" ", "s"), ("s", " "), (" ", "f"), ("f", "o"), ("o", "r"),
            ("r", " "), (" ", "w"), ("w", "i"), ("i", "t"), ("t", "h"), ("h", " "),
            (" ", "b"), ("b", "e"), ("e", " "), (" ", "y"), ("y", "o"), ("o", "u"),
            ("u", " "), (" ", "c"), ("c", "a"), ("a", "n"), ("n", " "), (" ", "d"),
            ("d", "o"), ("o", " "), (" ", "h"), ("h", "a"), ("a", "v"), ("v", "e"),
            ("e", " "), (" ", "i"), ("i", "t"), ("t", " "), (" ", "t"), ("t", "h"),
            ("h", "a"), ("a", "t"), ("t", " "), (" ", "b"), ("b", "y"), ("y", " "),
            (" ", "o"), ("o", "f"), ("f", " "), (" ", "t"), ("t", "h"), ("h", "i"),
            ("i", "s"), ("s", " "), (" ", "a"), ("a", "as"), ("s", " "), (" ", "w"),
            ("w", "e"), ("e", "r"), ("r", "e"), ("e", " "), (" ", "t"), ("t", "o"),
            ("o", " "), (" ", "b"), ("b", "e"), ("e", " "), (" ", "o"), ("o", "r"),
            ("r", " "), (" ", "n"), ("n", "o"), ("o", "t"), ("t", " "), (" ", "w"),
            ("w", "h"), ("h", "i"), ("i", "c"), ("c", "h"), ("h", " "), (" ", "a"),
            ("a", "r"), ("r", "e"), ("e", " "), (" ", "t"), ("t", "h"), ("h", "e"),
            ("e", "y"), ("y", " "), (" ", "w"), ("w", "e"), ("e", "r"), ("r", "e"),
            ("e", " "), (" ", "t"), ("t", "h"), ("h", "e"), ("e", "m"), ("m", " "),
            (" ", "a"), ("a", "n"), ("n", "d"), ("d", " "), (" ", "t"), ("t", "h"),
            ("h", "e"), ("e", "i"), ("i", "r"), ("r", " "), (" ", "o"), ("o", "f"),
            ("f", " "), (" ", "t"), ("t", "h"), ("h", "e"), ("e", " "),
            ("in", "g"), ("ed", " "), ("ly", " "), ("ti", "o"), ("al", " "),
            ("men", "t"), ("nes", "s"), ("ful", " "), ("les", "s"),
            ("=", "="), ("!", "="), ("<", "="), (">", "="), ("+", "="), ("-", "="),
            ("*", "="), ("/", "="), ("&", "&"), ("|", "|"), ("+", "+"), ("-", "-"),
            ("<", "<"), (">", ">"), ("(", ")"), ("[", "]"), ("{", "}"),
        ]

    def _get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        pairs = []
        if len(word) < 2: return pairs
        prev = word[0]
        for c in word[1:]:
            pairs.append((prev, c))
            prev = c
        return pairs

    def _bpe(self, token: str) -> List[str]:
        if token in self.vocab:
            return [token]
        word = list(token)
        while len(word) > 1:
            pairs = self._get_pairs(word)
            best_pair, best_pri = None, -1
            for p in pairs:
                if p in self.merge_lookup and self.merge_lookup[p] > best_pri:
                    best_pri = self.merge_lookup[p]
                    best_pair = p
            if best_pair is None:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    merged = word[i] + word[i + 1]
                    if merged in self.vocab:
                        new_word.append(merged)
                    else:
                        new_word.extend([word[i], word[i + 1]])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == len(new_word):
                break
        return word

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for m in self.pattern.finditer(text):
            t = m.group()
            if t.strip():
                tokens.extend(self._bpe(t))
        if tokens:
            tokens = [self.inverse_vocab[2]] + tokens + [self.inverse_vocab[3]]
        return tokens

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(t, self.special_tokens["<unk>"]) for t in self.tokenize(text)]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, "<unk>") for tid in token_ids]
        text = "".join(tokens)
        for st in self.special_tokens:
            text = text.replace(st, "")
        return text

    def tokenize_numbers_r2l(self, text: str) -> List[str]:
        tokens = []
        num_pat = re.compile(r'\d+\.?\d*')
        last = 0
        for m in num_pat.finditer(text):
            if m.start() > last:
                tokens.extend(self.tokenize(text[last:m.start()]))
            ns = m.group()
            if '.' in ns:
                ip, dp = ns.split('.')
                tokens.extend(
                    ['<num>'] + list(reversed(ip)) + ['<dec>'] + list(reversed(dp)) + ['</num>']
                )
            else:
                tokens.extend(['<num>'] + list(reversed(ns)) + ['</num>'])
            last = m.end()
        if last < len(text):
            tokens.extend(self.tokenize(text[last:]))
        return tokens

    def get_vocab_size(self) -> int:
        return len(self.vocab)


# ============================================================
# Shared activation functions
# ============================================================
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 +-*/=().%^<>,!?&|~@#$:;\"'\\/\n\t"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)
MAX_SEQ_LEN = 256


def text_to_char_ids(text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    text = text.lower()[:max_len]
    ids = np.zeros(max_len, dtype=np.int32)
    for i, c in enumerate(text):
        if c in CHAR_TO_IDX:
            ids[i] = CHAR_TO_IDX[c]
    return ids


def char_ids_to_embedding(char_ids: np.ndarray, embed_matrix: np.ndarray) -> np.ndarray:
    return embed_matrix[char_ids]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Correct analytical derivative of GELU."""
    k = 0.7978845608
    tanh_arg = k * (x + 0.044715 * x ** 3)
    tanh_arg = np.clip(tanh_arg, -20, 20)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1.0 - tanh_val ** 2
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * k * (1.0 + 3.0 * 0.044715 * x ** 2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sigmoid_grad(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-10)


def clip_grads(*grads, clip: float = 1.0):
    for g in grads:
        np.clip(g, -clip, clip, out=g)


# ============================================================
# NeuralMathDetector
# ============================================================
class NeuralMathDetector:
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        # 1. HE INITIALIZATION: Prevents the "0.25 Loss Plateau" by stabilizing starting variance
        self.char_embedding = (np.random.randn(CHAR_VOCAB_SIZE, embed_dim) * np.sqrt(2.0 / embed_dim)).astype(np.float32)
        
        # 2. PARAMETER BOOST: Input is (embed_dim * 2) because we concatenate Mean and Max pooling
        # Increased hidden layer to 64 for higher logic capacity
        self.fc1_w = (np.random.randn(embed_dim * 2, 64) * np.sqrt(2.0 / (embed_dim * 2))).astype(np.float32)
        self.fc1_b = np.zeros(64, dtype=np.float32)
        
        self.fc2_w = (np.random.randn(64, 1) * np.sqrt(2.0 / 64)).astype(np.float32)
        self.fc2_b = np.zeros(1, dtype=np.float32)

    def forward(self, text: str) -> float:
        # Pre-processing
        char_ids = text_to_char_ids(text)
        if len(char_ids) == 0: return 0.0
        
        # Embedding Look-up
        embeds = self.char_embedding[char_ids]
        
        # 3. ARCHITECTURE FIX: MEAN-MAX POOLING
        # np.mean captures the general "texture" (is it English text?)
        # np.max captures the "spikes" (is there a digit or math operator?)
        p_mean = np.mean(embeds, axis=0)
        p_max = np.max(embeds, axis=0)
        pooled = np.concatenate([p_mean, p_max]) 
        
        # Inference layers
        z1 = cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b
        h1 = gelu(z1)
        z2 = cpuwarp_ml.matmul(h1, self.fc2_w) + self.fc2_b
        
        # 4. NUMERICAL GUARD: Stable Sigmoid
        return float(1.0 / (1.0 + np.exp(-np.clip(z2[0], -15, 15))))

    def is_math_query(self, text: str, threshold: float = 0.5) -> bool:
        return self.forward(text) >= threshold

    def get_weights(self) -> Dict:
        return {
            "char_embedding": self.char_embedding.copy(),
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()


# ============================================================
# NeuralTypeClassifier
# ============================================================
class NeuralTypeClassifier:
    TYPE_NAMES = ["Arithmetic", "Algebraic", "Comparison", "Geometric", "Unknown"]
    NUM_TYPES = 5

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc1_w = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.1
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, self.NUM_TYPES).astype(np.float32) * 0.1
        self.fc2_b = np.zeros(self.NUM_TYPES, dtype=np.float32)

    def forward(self, text: str) -> np.ndarray:
        char_ids = text_to_char_ids(text)
        embeds = char_ids_to_embedding(char_ids, self.char_embedding)
        pooled = embeds.mean(axis=0)
        z1 = cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b
        h1 = gelu(z1)
        logits = cpuwarp_ml.matmul(h1, self.fc2_w) + self.fc2_b
        return softmax(logits)

    def classify(self, text: str) -> str:
        return self.TYPE_NAMES[int(np.argmax(self.forward(text)))]

    def get_probs(self, text: str) -> Dict[str, float]:
        probs = self.forward(text)
        return {n: float(probs[i]) for i, n in enumerate(self.TYPE_NAMES)}

    def get_weights(self) -> Dict:
        return {
            "char_embedding": self.char_embedding.copy(),
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy(); self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy(); self.fc2_b = w["fc2_b"].copy()


# ============================================================
# NeuralAimClassifier
# ============================================================
class NeuralAimClassifier:
    AIM_NAMES = ["Calculate", "Simplify", "Solve", "Compare", "Evaluate", "Unknown"]
    NUM_AIMS = 6

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc1_w = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.1
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, self.NUM_AIMS).astype(np.float32) * 0.1
        self.fc2_b = np.zeros(self.NUM_AIMS, dtype=np.float32)

    def forward(self, text: str) -> np.ndarray:
        char_ids = text_to_char_ids(text)
        embeds = char_ids_to_embedding(char_ids, self.char_embedding)
        pooled = embeds.mean(axis=0)
        z1 = cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b
        h1 = gelu(z1)
        logits = cpuwarp_ml.matmul(h1, self.fc2_w) + self.fc2_b
        return softmax(logits)

    def identify(self, text: str) -> str:
        return self.AIM_NAMES[int(np.argmax(self.forward(text)))]

    def get_probs(self, text: str) -> Dict[str, float]:
        probs = self.forward(text)
        return {n: float(probs[i]) for i, n in enumerate(self.AIM_NAMES)}

    def get_weights(self) -> Dict:
        return {
            "char_embedding": self.char_embedding.copy(),
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy(); self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy(); self.fc2_b = w["fc2_b"].copy()


# ============================================================
# NeuralSymbolicRouter
# ============================================================
class NeuralSymbolicRouter:
    ENGINE_NAMES = ["Native_Compute_Engine", "SymPy_Engine"]

    def __init__(self):
        self.fc_w = np.random.randn(11, 16).astype(np.float32) * 0.1
        self.fc_b = np.zeros(16, dtype=np.float32)
        self.out_w = np.random.randn(16, 2).astype(np.float32) * 0.1
        self.out_b = np.zeros(2, dtype=np.float32)

    def forward(self, type_probs: np.ndarray, aim_probs: np.ndarray) -> np.ndarray:
        x = np.concatenate([type_probs, aim_probs])
        h = gelu(cpuwarp_ml.matmul(x, self.fc_w) + self.fc_b)
        logits = cpuwarp_ml.matmul(h, self.out_w) + self.out_b
        return softmax(logits)

    def route(self, type_probs: np.ndarray, aim_probs: np.ndarray) -> str:
        return self.ENGINE_NAMES[int(np.argmax(self.forward(type_probs, aim_probs)))]

    def get_weights(self) -> Dict:
        return {
            "fc_w": self.fc_w.copy(), "fc_b": self.fc_b.copy(),
            "out_w": self.out_w.copy(), "out_b": self.out_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.fc_w = w["fc_w"].copy(); self.fc_b = w["fc_b"].copy()
        self.out_w = w["out_w"].copy(); self.out_b = w["out_b"].copy()


# ============================================================
# NeuralMathFilter
# ============================================================
class NeuralMathFilter:
    def __init__(self, embed_dim: int = 32):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc_w = np.random.randn(embed_dim, 1).astype(np.float32) * 0.1
        self.fc_b = np.zeros(1, dtype=np.float32)

    def filter(self, text: str, threshold: float = 0.5) -> str:
        text = text.lower()
        char_ids = np.array([CHAR_TO_IDX.get(c, 0) for c in text], dtype=np.int32)
        embeds = self.char_embedding[char_ids]
        scores = sigmoid(embeds @ self.fc_w + self.fc_b).squeeze(-1)
        result = [c for c, score in zip(text, scores) if c == ' ' or score >= threshold]
        return re.sub(r'\s+', ' ', ''.join(result)).strip()

    def get_weights(self) -> Dict:
        return {
            "char_embedding": self.char_embedding.copy(),
            "fc_w": self.fc_w.copy(), "fc_b": self.fc_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc_w = w["fc_w"].copy(); self.fc_b = w["fc_b"].copy()


# ============================================================
# NeuralOperatorMapper
# ============================================================
class NeuralOperatorMapper:
    OPERATORS = ["+", "-", "*", "/", "=", "^", "%", ">", "<", "**"]
    KNOWN_WORDS = [
        "plus", "add", "sum", "added", "minus", "subtract", "difference", "less", "spends",
        "times", "multiply", "product", "multiplied", "divided", "divide", "quotient", "over",
        "power", "raised", "squared", "cubed", "mod", "modulo", "remainder",
        "equals", "equal", "is", "gives", "greater", "gt", "lt",
    ]

    def __init__(self, embed_dim: int = 32):
        self.embed_dim = embed_dim
        self.word_embeddings = {
            w: np.random.randn(embed_dim).astype(np.float32) * 0.1 for w in self.KNOWN_WORDS
        }
        self.op_embeddings = np.random.randn(len(self.OPERATORS), embed_dim).astype(np.float32) * 0.1
        self.proj_w = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
        self.proj_b = np.zeros(embed_dim, dtype=np.float32)

    def _word_to_vec(self, word: str) -> np.ndarray:
        word = word.lower().strip()
        if word in self.word_embeddings:
            return self.word_embeddings[word]
        import hashlib
        idx = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % CHAR_VOCAB_SIZE
        return np.random.RandomState(idx).randn(self.embed_dim).astype(np.float32) * 0.1

    def map_word(self, word: str) -> str:
        word = word.lower().strip()
        if word in "+-*/=^%><":
            return word
        w_vec = self._word_to_vec(word)
        projected = gelu(cpuwarp_ml.matmul(w_vec, self.proj_w) + self.proj_b)
        return self.OPERATORS[int(np.argmax(cpuwarp_ml.matmul(projected, self.op_embeddings.T)))]

    def apply_to_text(self, text: str) -> str:
        words = re.findall(r'[a-zA-Z]+|[^\s]+', text.lower())
        result = [self.map_word(w) if re.match(r'^[a-z]+$', w) else w for w in words]
        return re.sub(r'\s+', ' ', ' '.join(result)).strip()

    def get_weights(self) -> Dict:
        return {
            "op_embeddings": self.op_embeddings.copy(),
            "proj_w": self.proj_w.copy(), "proj_b": self.proj_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.op_embeddings = w["op_embeddings"].copy()
        self.proj_w = w["proj_w"].copy(); self.proj_b = w["proj_b"].copy()


# ============================================================
# NeuralArithmeticSolver
# ============================================================
class NeuralArithmeticSolver:
    def __init__(self, hidden_dim: int = 128):
        self.fc1_w = np.random.randn(6, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 64).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(64, dtype=np.float32)
        self.fc3_w = np.random.randn(64, 1).astype(np.float32) * 0.01
        self.fc3_b = np.zeros(1, dtype=np.float32)
        self._answer_scale = 1.0

    def _extract_features(self, expression: str) -> Optional[np.ndarray]:
        cleaned = re.sub(r'\s+', '', expression).replace('^', '**')
        match = re.match(r'^(-?\d+\.?\d*)([\+\-\*/\*\*])(-?\d+\.?\d*)$', cleaned)
        if not match:
            return None
        n1_str, op, n2_str = match.groups()
        try:
            n1, n2 = float(n1_str), float(n2_str)
        except ValueError:
            return None
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3, '**': 4}
        oh = np.zeros(5)
        oh[op_map.get(op, 0)] = 1.0
        return np.array([n1, n2, oh[0], oh[1], oh[2], oh[3]], dtype=np.float32)

    def predict(self, expression: str) -> Optional[float]:
        features = self._extract_features(expression)
        if features is None:
            return None
        scale = self._answer_scale
        features = features.copy()
        features[0] /= (scale + 1e-8)
        features[1] /= (scale + 1e-8)
        h1 = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        h2 = gelu(cpuwarp_ml.matmul(h1, self.fc2_w) + self.fc2_b)
        raw = float((cpuwarp_ml.matmul(h2, self.fc3_w) + self.fc3_b)[0])
        return raw * scale

    def solve(self, expression: str) -> Optional[str]:
        result = self.predict(expression)
        if result is None:
            return None
        return str(int(round(result))) if abs(result - round(result)) < 0.01 else f"{result:.6g}"

    def get_weights(self) -> Dict:
        return {
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
            "fc3_w": self.fc3_w.copy(), "fc3_b": self.fc3_b.copy(),
            "_answer_scale": np.array([self._answer_scale]),
        }

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy(); self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy(); self.fc2_b = w["fc2_b"].copy()
        self.fc3_w = w["fc3_w"].copy(); self.fc3_b = w["fc3_b"].copy()
        if "_answer_scale" in w:
            self._answer_scale = float(w["_answer_scale"][0])


# ============================================================
# NeuralAlgebraicSolver
# ============================================================
class NeuralAlgebraicSolver:
    def __init__(self, hidden_dim: int = 64):
        self.fc1_w = np.random.randn(3, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 32).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(32, dtype=np.float32)
        self.fc3_w = np.random.randn(32, 1).astype(np.float32) * 0.01
        self.fc3_b = np.zeros(1, dtype=np.float32)

    def _parse_linear(self, expression: str) -> Optional[Tuple[float, float, float]]:
        match = re.match(
            r'([\d\.\+\-\*/\*\s]*)\s*([a-zA-Z])\s*([\+\-\d\.\*/\s]*)\s*=\s*([\d\.\+\-\*/\s]*)',
            expression,
        )
        if not match:
            return None
        try:
            lhs_coef, var, lhs_const, rhs = match.groups()
            lhs_coef = lhs_coef.strip()
            if not lhs_coef or lhs_coef in ('+', '-'):
                coef = 1.0 if not lhs_coef or lhs_coef == '+' else -1.0
            else:
                coef = float(eval(lhs_coef, {"__builtins__": {}}, {}))
            const = float(eval(lhs_const.strip(), {"__builtins__": {}}, {})) if lhs_const.strip() else 0.0
            rhs_val = float(eval(rhs.strip(), {"__builtins__": {}}, {}))
            return (coef, const, rhs_val)
        except Exception:
            return None

    def predict(self, expression: str) -> Optional[float]:
        parsed = self._parse_linear(expression)
        if parsed is None:
            return None
        features = np.array(parsed, dtype=np.float32)
        h1 = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        h2 = gelu(cpuwarp_ml.matmul(h1, self.fc2_w) + self.fc2_b)
        return float((cpuwarp_ml.matmul(h2, self.fc3_w) + self.fc3_b)[0])

    def solve(self, expression: str) -> Optional[str]:
        result = self.predict(expression)
        if result is None:
            return None
        return f"x = {int(round(result))}" if abs(result - round(result)) < 0.01 else f"x = {result:.6g}"

    def get_weights(self) -> Dict:
        return {
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
            "fc3_w": self.fc3_w.copy(), "fc3_b": self.fc3_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy(); self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy(); self.fc2_b = w["fc2_b"].copy()
        self.fc3_w = w["fc3_w"].copy(); self.fc3_b = w["fc3_b"].copy()


# ============================================================
# NeuralComparisonSolver
# ============================================================
class NeuralComparisonSolver:
    def __init__(self, hidden_dim: int = 32):
        self.fc1_w = np.random.randn(2, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 3).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(3, dtype=np.float32)

    def predict(self, a: float, b: float) -> np.ndarray:
        features = np.array([a, b], dtype=np.float32)
        h = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        logits = cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b
        return softmax(logits)

    def solve(self, expression: str) -> Optional[str]:
        match = re.search(r'(\d+\.?\d*)\s*([><=!]+)\s*(\d+\.?\d*)', expression)
        if match:
            left, op, right = float(match.group(1)), match.group(2), float(match.group(3))
            probs = self.predict(left, right)
            ops = [">", "<", "="]
            return f"{left} {ops[int(np.argmax(probs))]} {right} ({probs.max():.3f})"
        match = re.search(r'compare\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)', expression, re.IGNORECASE)
        if match:
            left, right = float(match.group(1)), float(match.group(2))
            probs = self.predict(left, right)
            ops = [">", "<", "="]
            return f"{left} {ops[int(np.argmax(probs))]} {right} ({probs.max():.3f})"
        return None

    def get_weights(self) -> Dict:
        return {
            "fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
            "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
        }

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy(); self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy(); self.fc2_b = w["fc2_b"].copy()


# ============================================================
# Unified NeuralMVModel
# ============================================================
class NeuralMVModel:
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        self.detector = NeuralMathDetector(embed_dim)
        self.type_classifier = NeuralTypeClassifier(embed_dim, hidden_dim)
        self.aim_classifier = NeuralAimClassifier(embed_dim, hidden_dim)
        self.symbolic_router = NeuralSymbolicRouter()
        self.math_filter = NeuralMathFilter(embed_dim)
        self.op_mapper = NeuralOperatorMapper(embed_dim)
        self.arith_solver = NeuralArithmeticSolver(hidden_dim)
        self.algebra_solver = NeuralAlgebraicSolver(hidden_dim)
        self.comparison_solver = NeuralComparisonSolver(hidden_dim)
        self.solve_history = []

    def is_math_query(self, text: str) -> bool:
        return self.detector.is_math_query(text)

    def math_confidence(self, text: str) -> float:
        return self.detector.forward(text)

    def solve(self, query: str) -> Dict:
        problem_type = self.type_classifier.classify(query)
        aim = self.aim_classifier.identify(query)
        type_probs = self.type_classifier.forward(query)
        aim_probs = self.aim_classifier.forward(query)
        engine = self.symbolic_router.route(type_probs, aim_probs)

        result = None
        if engine == "Native_Compute_Engine":
            if problem_type == "Comparison" or aim == "Compare":
                result = self.comparison_solver.solve(query)
            if result is None:
                cleaned = self.math_filter.filter(query)
                symbolic = self.op_mapper.apply_to_text(cleaned)
                result = self.arith_solver.solve(symbolic)
        elif engine == "SymPy_Engine":
            if problem_type == "Comparison":
                result = self.comparison_solver.solve(query)
            if result is None:
                result = self.algebra_solver.solve(query)
            if result is None:
                cleaned = self.math_filter.filter(query)
                symbolic = self.op_mapper.apply_to_text(cleaned)
                result = self.arith_solver.solve(symbolic)

        if result is None:
            result = "Unable to solve"

        solution = {
            "query": query, "problem_type": problem_type, "aim": aim,
            "engine": engine, "result": result,
            "tokenized": {
                "original": query,
                "cleaned": self.math_filter.filter(query),
                "symbolic": self.op_mapper.apply_to_text(query),
            },
            "type_probs": {k: float(v) for k, v in self.type_classifier.get_probs(query).items()},
            "aim_probs":  {k: float(v) for k, v in self.aim_classifier.get_probs(query).items()},
        }
        self.solve_history.append(solution)
        return solution

    def get_all_weights(self) -> Dict:
        return {
            "math_detector":     self.detector.get_weights(),
            "type_classifier":   self.type_classifier.get_weights(),
            "aim_classifier":    self.aim_classifier.get_weights(),
            "symbolic_router":   self.symbolic_router.get_weights(),
            "math_filter":       self.math_filter.get_weights(),
            "op_mapper":         self.op_mapper.get_weights(),
            "arith_solver":      self.arith_solver.get_weights(),
            "algebra_solver":    self.algebra_solver.get_weights(),
            "comparison_solver": self.comparison_solver.get_weights(),
        }

    def load_all_weights(self, weights: Dict):
        self.detector.load_weights(weights.get("math_detector", {}))
        self.type_classifier.load_weights(weights.get("type_classifier", {}))
        self.aim_classifier.load_weights(weights.get("aim_classifier", {}))
        self.symbolic_router.load_weights(weights.get("symbolic_router", {}))
        self.math_filter.load_weights(weights.get("math_filter", {}))
        self.op_mapper.load_weights(weights.get("op_mapper", {}))
        self.arith_solver.load_weights(weights.get("arith_solver", {}))
        self.algebra_solver.load_weights(weights.get("algebra_solver", {}))
        self.comparison_solver.load_weights(weights.get("comparison_solver", {}))

    def count_weights(self) -> int:
        return sum(
            arr.size for cw in self.get_all_weights().values()
            for arr in cw.values() if isinstance(arr, np.ndarray)
        )


# ============================================================
# NeuralMVTrainer
# ============================================================
class NeuralMVTrainer:
    def __init__(self, model: NeuralMVModel, lr: float = 0.001):
        self.model = model
        self.lr = lr

    def _embed(self, text: str, embed_matrix: np.ndarray):
        char_ids = text_to_char_ids(text)
        embeds = embed_matrix[char_ids]
        pooled = embeds.mean(axis=0)
        return pooled, embeds, char_ids

    def train_detector(self, texts: List[str], labels: List[int], epochs: int = 10):
        d = self.model.detector
        for epoch in range(epochs):
            total_loss = 0.0
            idxs = list(range(len(texts)))
            random.shuffle(idxs)
            for i in idxs:
                text, label = texts[i], labels[i]
                pooled, _, _ = self._embed(text, d.char_embedding)
                z1 = pooled @ d.fc1_w + d.fc1_b
                h1 = gelu(z1)
                z2 = h1 @ d.fc2_w + d.fc2_b
                pred = sigmoid(z2)
                err = pred - label
                loss = float((err ** 2).sum())
                total_loss += loss

                dz2 = 2.0 * err * sigmoid_grad(pred)
                grad_fc2_w = np.outer(h1, dz2)
                grad_fc2_b = dz2
                dh1 = dz2 @ d.fc2_w.T
                dz1 = dh1 * gelu_grad(z1)
                grad_fc1_w = np.outer(pooled, dz1)
                grad_fc1_b = dz1

                clip_grads(grad_fc2_w, grad_fc2_b, grad_fc1_w, grad_fc1_b)

                d.fc2_w -= self.lr * grad_fc2_w
                d.fc2_b -= self.lr * grad_fc2_b
                d.fc1_w -= self.lr * grad_fc1_w
                d.fc1_b -= self.lr * grad_fc1_b
            print(f"  Detector epoch {epoch+1}/{epochs}, loss: {total_loss/len(texts):.4f}", flush=True)

    def train_type_classifier(self, texts: List[str], labels: List[int], epochs: int = 10):
        tc = self.model.type_classifier
        for epoch in range(epochs):
            total_loss = 0.0
            idxs = list(range(len(texts)))
            random.shuffle(idxs)
            for i in idxs:
                text, label = texts[i], labels[i]
                pooled, _, _ = self._embed(text, tc.char_embedding)
                z1 = pooled @ tc.fc1_w + tc.fc1_b
                h1 = gelu(z1)
                logits = h1 @ tc.fc2_w + tc.fc2_b
                probs = softmax(logits)

                target = np.zeros(tc.NUM_TYPES, dtype=np.float32)
                target[label] = 1.0
                loss = -float(np.sum(target * np.log(probs + 1e-10)))
                total_loss += loss

                dlogits = probs - target
                grad_fc2_w = np.outer(h1, dlogits)
                grad_fc2_b = dlogits
                dh1 = dlogits @ tc.fc2_w.T
                dz1 = dh1 * gelu_grad(z1)
                grad_fc1_w = np.outer(pooled, dz1)
                grad_fc1_b = dz1

                clip_grads(grad_fc2_w, grad_fc2_b, grad_fc1_w, grad_fc1_b)

                tc.fc2_w -= self.lr * grad_fc2_w
                tc.fc2_b -= self.lr * grad_fc2_b
                tc.fc1_w -= self.lr * grad_fc1_w
                tc.fc1_b -= self.lr * grad_fc1_b
            print(f"  Type epoch {epoch+1}/{epochs}, loss: {total_loss/len(texts):.4f}", flush=True)

    def train_aim_classifier(self, texts: List[str], labels: List[int], epochs: int = 10):
        ac = self.model.aim_classifier
        for epoch in range(epochs):
            total_loss = 0.0
            for text, label in zip(texts, labels):
                pooled, _, _ = self._embed(text, ac.char_embedding)
                z1 = pooled @ ac.fc1_w + ac.fc1_b
                h1 = gelu(z1)
                logits = h1 @ ac.fc2_w + ac.fc2_b
                probs = softmax(logits)

                target = np.zeros(ac.NUM_AIMS, dtype=np.float32)
                target[label] = 1.0
                loss = -float(np.sum(target * np.log(probs + 1e-10)))
                total_loss += loss

                dlogits = probs - target
                grad_fc2_w = np.outer(h1, dlogits)
                grad_fc2_b = dlogits
                dh1 = dlogits @ ac.fc2_w.T
                dz1 = dh1 * gelu_grad(z1)
                grad_fc1_w = np.outer(pooled, dz1)
                grad_fc1_b = dz1

                clip_grads(grad_fc2_w, grad_fc2_b, grad_fc1_w, grad_fc1_b)

                ac.fc2_w -= self.lr * grad_fc2_w
                ac.fc2_b -= self.lr * grad_fc2_b
                ac.fc1_w -= self.lr * grad_fc1_w
                ac.fc1_b -= self.lr * grad_fc1_b
            print(f"  Aim epoch {epoch+1}/{epochs}, loss: {total_loss/len(texts):.4f}", flush=True)

    def train_arith_solver(self, expressions: List[str], answers: List[float], epochs: int = 10):
        s = self.model.arith_solver
        max_ans = max(abs(a) for a in answers) + 1e-8
        s._answer_scale = max_ans
        norm_answers = [a / max_ans for a in answers]

        for epoch in range(epochs):
            total_loss, count = 0.0, 0
            for expr, norm_answer in zip(expressions, norm_answers):
                raw_features = s._extract_features(expr)
                if raw_features is None: continue
                features = raw_features.copy()
                features[0] /= (max_ans + 1e-8)
                features[1] /= (max_ans + 1e-8)

                z1 = features @ s.fc1_w + s.fc1_b
                h1 = gelu(z1)
                z2 = h1 @ s.fc2_w + s.fc2_b
                h2 = gelu(z2)
                z3 = h2 @ s.fc3_w + s.fc3_b
                pred = z3[0]
                err = pred - norm_answer
                loss = err ** 2
                total_loss += loss
                count += 1

                dz3 = np.array([2.0 * err], dtype=np.float32)
                grad_fc3_w = np.outer(h2, dz3)
                grad_fc3_b = dz3
                dh2 = dz3 @ s.fc3_w.T
                dz2 = dh2 * gelu_grad(z2)
                grad_fc2_w = np.outer(h1, dz2)
                grad_fc2_b = dz2
                dh1 = dz2 @ s.fc2_w.T
                dz1 = dh1 * gelu_grad(z1)
                grad_fc1_w = np.outer(features, dz1)
                grad_fc1_b = dz1

                clip_grads(grad_fc3_w, grad_fc3_b, grad_fc2_w, grad_fc2_b, grad_fc1_w, grad_fc1_b, clip=1.0)

                s.fc3_w -= self.lr * grad_fc3_w
                s.fc3_b -= self.lr * grad_fc3_b
                s.fc2_w -= self.lr * grad_fc2_w
                s.fc2_b -= self.lr * grad_fc2_b
                s.fc1_w -= self.lr * grad_fc1_w
                s.fc1_b -= self.lr * grad_fc1_b
            if count > 0:
                print(f"  Arith epoch {epoch+1}/{epochs}, loss: {total_loss/count:.6f}", flush=True)


# ============================================================
# Neural Orchestration System
# ============================================================
class NeuralOrchestrationSystem:
    def __init__(self, num_workers: int = 8, num_neurons: int = 16, max_retries: int = 4, d_model: int = 1024):
        self.num_workers = num_workers
        self.num_neurons = num_neurons
        self.max_retries = min(max_retries, num_neurons)
        self.d_model = d_model
        self._init_components()
        self.metrics = {
            "worker_outputs": 0, "orchestrator_scores": 0,
            "manager_routing_decisions": 0, "safety_filter_activations": 0,
            "verifier_acceptances": 0, "verifier_rejections": 0,
            "retry_attempts": 0, "retry_successes": 0, "unsafe_content_blocked": 0,
        }
        print(f"Neural Orchestration: {num_workers} workers, {num_neurons} neurons, {max_retries} retries")

    def _init_components(self):
        self.worker_nodes = [
            {"weights": np.random.randn(self.d_model, self.d_model).astype(np.float32) * 0.02,
             "bias": np.random.randn(self.d_model).astype(np.float32) * 0.02, "activation": "gelu"}
            for _ in range(self.num_workers)
        ]
        self.orchestrator = {
            "scoring_weights": np.random.randn(self.d_model, self.num_neurons).astype(np.float32) * 0.01,
            "routing_weights": np.random.randn(self.d_model, self.num_neurons).astype(np.float32) * 0.01,
            "composite_weights": np.random.randn(self.num_neurons * 2, 1).astype(np.float32) * 0.01,
        }
        self.manager_node = {"decision_threshold": 0.7, "selection_weights": np.random.randn(self.num_neurons, 1).astype(np.float32) * 0.01}
        self.safety_guardrail = {
            "query_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32) * 0.02,
            "key_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32) * 0.02,
            "value_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32) * 0.02,
            "bad_matrices": np.random.randn(self.d_model, 10).astype(np.float32) * 0.1,
            "safety_threshold": 0.8,
        }
        self.verifier = {"normalization_factor": 1.0, "aggregation_weights": np.random.randn(4, 1).astype(np.float32) * 0.01, "acceptance_threshold": 0.3}
        self.retry_policy = {"retry_counter": 0, "max_retries": self.max_retries, "retry_decay": 0.9}

    def get_state(self) -> dict:
        return {
            "num_workers": self.num_workers, "num_neurons": self.num_neurons,
            "max_retries": self.max_retries, "d_model": self.d_model,
            "worker_nodes": [{"weights": n["weights"].copy(), "bias": n["bias"].copy(), "activation": n["activation"]} for n in self.worker_nodes],
            "orchestrator": {k: v.copy() for k, v in self.orchestrator.items()},
            "manager_node": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.manager_node.items()},
            "safety_guardrail": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.safety_guardrail.items()},
            "verifier": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.verifier.items()},
            "retry_policy": dict(self.retry_policy), "orchestration_metrics": dict(self.metrics),
        }

    def restore_state(self, state: dict):
        self.num_workers = state["num_workers"]; self.num_neurons = state["num_neurons"]
        self.max_retries = state["max_retries"]; self.d_model = state["d_model"]
        self.worker_nodes = state["worker_nodes"]
        self.orchestrator = {k: v.copy() for k, v in state["orchestrator"].items()}
        self.manager_node = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state["manager_node"].items()}
        self.safety_guardrail = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state["safety_guardrail"].items()}
        self.verifier = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state["verifier"].items()}
        self.retry_policy = dict(state["retry_policy"])
        if "orchestration_metrics" in state: self.metrics.update(state["orchestration_metrics"])


# ============================================================
# ScavengerDataset
# ============================================================
class ScavengerDataset:
    def __init__(self, max_size: int = 8000, min_quality: float = 0.7, auto_scavenge: bool = True, dataset_path: Optional[str] = None):
        self.max_size = max_size; self.min_quality = min_quality
        self.samples: List[str] = []; self.sources_used: List[str] = []; self.quality_scores: List[float] = []
        if dataset_path: self._load_path(dataset_path)
        elif auto_scavenge: self._auto_find_math_json()

    def _auto_find_math_json(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [base_dir, os.path.join(base_dir, "data"), os.path.join(base_dir, "datasets")]
        for sd in search_dirs:
            if os.path.isdir(sd):
                for f in os.listdir(sd):
                    if f.endswith('.json') and ('math' in f.lower() or 'synthetic' in f.lower()):
                        fp = os.path.join(sd, f); self._load_json(fp); self.sources_used.append(fp)
                        print(f"  Auto-found: {fp} ({len(self.samples)} samples)"); return
        print("  No math JSON found. Generating synthetic dataset..."); self._generate_synthetic()

    def _load_path(self, path: str):
        if os.path.exists(path): self._load_json(path); self.sources_used.append(path); print(f"  Loaded: {path} ({len(self.samples)} samples)")
        else: print(f"  Path not found: {path}. Generating synthetic..."); self._generate_synthetic()

    def _load_json(self, path: str):
        with open(path, 'r') as f: data = json.load(f)
        for p in data.get("problems", [])[:self.max_size]: self.samples.append(f"<math>{p['problem']}</math>"); self.quality_scores.append(0.95)

    def _generate_synthetic(self):
        # Placeholder for actual generation logic
        for _ in range(self.max_size): self.samples.append("<math>1+1</math>"); self.quality_scores.append(0.95)

    def get_sample_count(self) -> int: return len(self.samples)
    def get_samples(self) -> List[str]: return self.samples


# ============================================================
# Checkpoint Manager
# ============================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

def find_latest_checkpoint() -> Optional[str]:
    if not os.path.isdir(CHECKPOINT_DIR): return None
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith("_mv_weights.pkl")]
    if not ckpts: return None
    ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    return os.path.join(CHECKPOINT_DIR, ckpts[0])

def load_checkpoint_state() -> Optional[Dict]:
    state_path = os.path.join(CHECKPOINT_DIR, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f: return json.load(f)
    return None

def save_checkpoint(model: NeuralMVModel, total_epochs: int, dataset_name: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    mv_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_mv_weights.pkl")
    with open(mv_path, "wb") as f: pickle.dump(model.get_all_weights(), f)
    state = {"total_epochs": total_epochs, "dataset": dataset_name, "checkpoint_file": mv_path, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(CHECKPOINT_DIR, "training_state.json"), "w") as f: json.dump(state, f, indent=2)
    print(f"Checkpoint saved: {mv_path} (epoch {total_epochs})")


# ============================================================
# Build type_map dynamically
# ============================================================
def build_type_map(problems: List[Dict]) -> Dict[str, int]:
    actual_cats = set(p.get("category", "unknown") for p in problems)
    base_map = {
        "Arithmetic": 0, "arithmetic": 0, "Algebra": 1, "Algebraic": 1,
        "Comparison": 2, "Geometry": 3, "GSM8K-Reasoning": 0, "None": 4, "unknown": 4
    }
    final_map = {}
    for cat in actual_cats: final_map[cat] = base_map.get(cat, 4)
    print(f"  Resolved type_map: {final_map}")
    return final_map


# ============================================================
# Training Pipeline
# ============================================================
def train_neural_mv(dataset_path: str = "synthetic_math_dataset.json", epochs: int = 5, lr: float = 0.01, resume: bool = True):
    print("=" * 60); print("Training Neural MV Pipeline"); print("=" * 60)
    with open(dataset_path, "r") as f: data = json.load(f)
    problems = data["problems"]; print(f"Loaded {len(problems)} math problems")
    model = NeuralMVModel(embed_dim=64, hidden_dim=128); start_epoch = 0
    if resume:
        ckpt = find_latest_checkpoint(); state = load_checkpoint_state()
        if ckpt and state:
            with open(ckpt, "rb") as f: weights = pickle.load(f)
            model.load_all_weights(weights); start_epoch = state.get("total_epochs", 0)
            print(f"Resuming from epoch {start_epoch}")
    trainer = NeuralMVTrainer(model, lr=lr); dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # ---- Phase 1: Math Detector ----
    print("\n[1/4] Training Math Detector...")
    math_problems = [p for p in problems if p.get("category") != "None"]
    non_math_problems = [p for p in problems if p.get("category") == "None"]
    n_math = min(500, len(math_problems))
    math_texts = [p["problem"] for p in math_problems[:n_math]]
    non_math_texts = [p["problem"] for p in non_math_problems[:n_math]] if non_math_problems else ["Hello", "World"] * (n_math // 2)
    trainer.train_detector(math_texts + non_math_texts, [1]*len(math_texts) + [0]*len(non_math_texts), epochs=epochs)

    # ---- Phase 2: Type Classifier ----
    print("\n[2/4] Training Type Classifier...")
    type_map = build_type_map(problems)
    type_texts = [p["problem"] for p in problems[:1000]]
    type_labels = [type_map.get(p.get("category", "unknown"), 4) for p in problems[:1000]]
    trainer.train_type_classifier(type_texts, type_labels, epochs=epochs)

    # ---- Phase 3: Arithmetic Solver ----
    print("\n[3/4] Training Arithmetic Solver...")
    arith_exprs = [p["problem"] for p in math_problems[:500]]
    arith_answers = [float(p.get("answer", 0)) for p in math_problems[:500]]
    if arith_exprs: trainer.train_arith_solver(arith_exprs, arith_answers, epochs=epochs)

    # ---- Phase 4: Save ----
    total_epochs = start_epoch + epochs
    save_checkpoint(model, total_epochs, dataset_name)
    return model

if __name__ == "__main__":
    dataset = ScavengerDataset(max_size=8000, auto_scavenge=True)
    dataset_path = dataset.sources_used[0] if dataset.sources_used else "math_data.json"
    # STABILIZED PRECISION SCHEDULE
    print("\n=== EXECUTING STABILIZED PRECISION SCHEDULE ===")
    print("\n--- PHASE 1: Magnitude Alignment (60 epochs @ LR=0.01) ---")
    model = train_neural_mv(dataset_path=dataset_path, epochs=60, lr=0.01, resume=False)
    print("\n--- PHASE 2: High-Speed Convergence (60 epochs @ LR=0.05) ---")
    model = train_neural_mv(dataset_path=dataset_path, epochs=60, lr=0.05, resume=True)
