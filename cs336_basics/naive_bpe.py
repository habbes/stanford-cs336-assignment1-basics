import regex as re

# See: https://github.com/openai/tiktoken/pull/234/files 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# pretokenize
# merges

class NaiveBPE:
    def __init__(self, corpus: str, num_merges: int):
        self.vocab = [
            chr(i).encode('utf-8') for i in range(256)
        ]
        self.vocab = ['<|endoftext|>'.encode('utf-8')] + self.vocab
        self.corpus = corpus
        self.num_merges = num_merges
        self.pretokenized_cache = None
        self.merged_cache = None

    def run(self):
        self._pretokenize()
        self._merge()

    def _pretokenize(self):
        pretokens = re.finditer(PAT, self.corpus)
        cache = {}
        for match in pretokens:
            token = match.group(0)
            encoded_token = token.encode("utf-8")
            token_key = tuple(encoded_token[i:i+1] for i in range(len(encoded_token)))
            if token_key not in cache:
                cache[token_key] = 0
            cache[token_key] += 1
        
        self.pretokenized_cache = cache
    
    
    def _merge(self):
        old_cache = self.pretokenized_cache
        for _ in range(self.num_merges):
            pair_counts = {}
            best_pair = None
            best_count = 0
            for token_key, count in self.pretokenized_cache.items():
                for i in range(len(token_key) - 1):
                    pair = (token_key[i], token_key[i + 1])
                    if pair not in pair_counts:
                        pair_counts[pair] = count
                    else:
                        pair_counts[pair] += count
                    
                    if best_pair is None:
                        best_pair = pair
                        best_count = pair_counts[pair]
                    elif pair_counts[pair] > best_count:
                        best_pair = pair
                        best_count = pair_counts[pair]
                    elif pair_counts[pair] == best_count:
                        if pair > best_pair:
                            best_pair = pair
                            best_count = pair_counts[pair]
            
            self.vocab.append(b"".join(best_pair))
            # merge the best pair in the pretokenized cache
            new_cache = {}
            for token_key, count in old_cache.items():
                temp_token_key = []
                i = 0
                while i < len(token_key):
                    if i == len(token_key) - 1:
                        temp_token_key.append(token_key[i])
                        i += 1
                        continue
                    if token_key[i] == best_pair[0] and token_key[i + 1] == best_pair[1]:
                        temp_token_key.append(best_pair[0] + best_pair[1])
                        i += 2
                        continue
                    
                    temp_token_key.append(token_key[i])
                    i += 1
                
                new_token_key = tuple(temp_token_key)
                new_cache[new_token_key] = count

            old_cache = new_cache
        self.merged_cache = old_cache

            
sample_text = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

bpe = NaiveBPE(sample_text, 6)
bpe.run()

print("vocab", bpe.vocab)

                    
                    