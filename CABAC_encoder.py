import time
import math
import numpy as np
from collections import defaultdict

class CABACEncoder:
    def __init__(self):
        self.prob_states = {0: 0.5, 1: 0.5, 2: 0.5}
        self.total_bits = 0
        self.bit_breakdown = defaultdict(float)
        self.encoding_time = 0
        self.symbol_bits = []
        
    def encode_sequence(self, symbols, contexts):
        start_time = time.time()
        self.symbol_bits = []
        for sym, ctx in zip(symbols, contexts):
            prob = self.prob_states[ctx]
            bits = -math.log2(prob if sym > 5 else 1-prob)
            self.total_bits += bits
            self.symbol_bits.append(bits)
            self.bit_breakdown[ctx] += bits
            
            if sym > 5:
                self.prob_states[ctx] += 0.1 * (1 - self.prob_states[ctx])
            else:
                self.prob_states[ctx] -= 0.1 * self.prob_states[ctx]
            self.prob_states[ctx] = np.clip(self.prob_states[ctx], 0.01, 0.99)
        
        self.encoding_time = time.time() - start_time
        return self.total_bits
