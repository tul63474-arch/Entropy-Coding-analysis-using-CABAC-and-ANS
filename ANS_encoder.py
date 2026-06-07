import time
import math
import numpy as np

class ANSEncoder:
    def __init__(self, scale=4096):
        self.scale = scale
        self.total_bits = 0
        self.encoding_time = 0
        self.symbol_bits = []
        self.freq_table = None
        self.cdf_table = None
        
    def build_model(self, symbols, num_symbols=11):
        hist = np.bincount(symbols, minlength=num_symbols)
        probs = hist / len(symbols)
        freqs = np.maximum(1, (probs * self.scale).astype(int))
        freqs[-1] += self.scale - np.sum(freqs)
        self.freq_table = freqs
        self.cdf_table = np.cumsum(freqs)
        self.total_freq = self.cdf_table[-1]
        
    def encode_sequence(self, symbols):
        start_time = time.time()
        self.symbol_bits = []
        
        if self.freq_table is None:
            self.build_model(symbols)
        
        state = self.total_freq
        for sym in reversed(symbols):
            freq = self.freq_table[sym]
            cdf = self.cdf_table[sym] - freq
            q = state // freq
            r = state % freq
            state = (q << 16) + cdf + r
            bits = math.log2(state) if state > 0 else 0
            self.symbol_bits.append(bits)
        
        self.total_bits = math.log2(state) if state > 0 else 0
        self.encoding_time = time.time() - start_time
        return self.total_bits
