# bpe_jax.py

import os
import json
import regex as re
import requests

import jax
import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
# The core Byte Pair Encoder logic remains the same (unchanged from your original).


def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # bpe token encoder/decoder
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # Regex pattern for pre-tokenization (same as original)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}

    def bpe(self, token):
        """
        Use self.bpe_ranks to iteratively merge BPE pairs in 'token' until no more merges.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram

            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        Convert a string into a list of integer BPE token IDs.
        """
        bpe_idx = []
        # pre-tokenize
        tokens = re.findall(self.pat, text)
        # process each piece
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(" ")
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """
        For debugging; returns more intermediate detail (pre-tokenization, merges, etc.).
        """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(" ")
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append(
                {
                    "token": token,
                    "token_bytes": token_bytes,
                    "token_translated": token_translated,
                    "token_merged": token_merged,
                    "token_ix": token_ix,
                }
            )
        out = {
            "bpe_idx": bpe_idx,
            "tokens": tokens,
            "parts": parts,
        }
        return out

    def decode(self, bpe_idx):
        """
        Convert a list of integer BPE token IDs back to a string.
        """
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = "".join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode("utf-8", errors="replace")
        return text


# -----------------------------------------------------------------------------
# Helpers for downloading the GPT-2 merges and vocab


def get_file(local_file, remote_file):
    """Downloads remote_file to local_file if it's not already present."""
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)


def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files under ~/.cache/mingpt/.
    """
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "mingpt")
    os.makedirs(cache_dir, exist_ok=True)

    # Load encoder.json that has the raw mappings from token -> bpe index
    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_file = (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
    )
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, "r") as f:
        encoder = json.load(f)
    assert (
        len(encoder) == 50257
    )  # GPT-2 has 50,000 merges + 256 byte tokens + 1 special token

    # Load vocab.bpe that contains the BPE merges
    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    vocab_remote_file = (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
    )
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    assert len(bpe_merges) == 50000

    return Encoder(encoder, bpe_merges)


# -----------------------------------------------------------------------------
# JAX-friendly wrapper


class BPETokenizerJax:
    """
    JAX-aware class that wraps the GPT-2 BPE Encoder.
    By default, returns jax.numpy arrays if you set return_tensors='jax'.
    """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors="jax"):
        """
        Tokenize text and return a batched JAX array of shape (batch_size, sequence_length).
        Args:
            text (str or list[str]): Input text(s) to tokenize.
            return_tensors (str): If 'jax', returns jax.numpy array.
                                  If 'np', returns plain numpy array.
        Returns:
            A jax.numpy array or numpy array of shape (batch_size, seq_length).
        """
        # Handle either a single string or a list of strings
        if isinstance(text, str):
            text = [text]  # wrap single string in a list

        assert isinstance(text, list), "Input must be a string or list of strings."

        # Encode each string to a list of BPE IDs
        encoded_batch = []
        for t in text:
            bpe_ids = self.encoder.encode(t)
            encoded_batch.append(bpe_ids)

        # We can store as a Ragged array, or pad if we want uniform shape:
        # For demonstration, let's just store them as a plain list-of-lists.
        # Convert to np array with dtype=int32 (ragged if lengths differ).
        # If you want a fixed block size, you could pad/truncate here.
        arr = [np.array(ids, dtype=np.int32) for ids in encoded_batch]

        # Optionally, we can do no padding and just keep the data as a list of arrays.
        # But if you want a single array, you'll need to decide on a padding length.
        # For now, let's assume no padding, just return a list of jnp arrays.
        # If you need a single fixed shape, you'd pad them to the same length.

        # Convert to the requested tensor type
        if return_tensors == "jax":
            arr = [jnp.array(a) for a in arr]
        elif return_tensors == "np":
            # Already a list of np arrays
            pass
        else:
            raise ValueError("Unsupported return_tensors type. Use 'jax' or 'np'.")

        # If you only want a single array (batch_size, seq_len), you'd have to pad here.
        # For example:
        # max_len = max(len(a) for a in arr)
        # padded = []
        # for a in arr:
        #     padded_arr = np.pad(a, (0, max_len - len(a)), constant_values=0)
        #     padded.append(padded_arr)
        # arr = np.stack(padded, axis=0)
        # if return_tensors == 'jax':
        #     arr = jnp.array(arr)

        return arr

    def decode(self, token_ids):
        """
        Decodes a sequence (or batch of sequences) of token IDs back to text.
        Args:
            token_ids (list[int] or np.ndarray or jnp.ndarray or list of those):
                Either a single token sequence or a list of token sequences.

        Returns:
            A single string if token_ids is 1D, or a list of strings if token_ids is a batch.
        """
        # If token_ids is a single sequence, convert it to Python list if needed:
        if isinstance(token_ids, (np.ndarray, jnp.ndarray)):
            # Check if 1D
            if token_ids.ndim == 1:
                # Single sequence
                return self.encoder.decode(token_ids.tolist())
            elif token_ids.ndim == 2:
                # Batch of sequences
                return [self.encoder.decode(seq.tolist()) for seq in token_ids]
            else:
                raise ValueError("decode only supports 1D or 2D arrays.")
        elif isinstance(token_ids, list):
            # Could be a single list of ints, or list of list-of-ints
            if len(token_ids) == 0:
                return ""
            if all(isinstance(x, int) for x in token_ids):
                # Single sequence
                return self.encoder.decode(token_ids)
            elif all(isinstance(x, list) for x in token_ids):
                # Batch
                return [self.encoder.decode(seq) for seq in token_ids]
            else:
                raise ValueError(
                    "decode expects a list of ints or list of list-of-ints."
                )
        else:
            raise ValueError("Unsupported token_ids type for decode.")


# -----------------------------------------------------------------------------


if __name__ == "__main__":

    # here is an encoding example
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(r["tokens"])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("Then we iterate over each chunk and process them in turn...")
    for part in r["parts"]:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (refer to the code inside Encoder.encode for what these intermediates are)
    print("and the final outcome is concatenating and flattening all the token_ix:")
    print(r["bpe_idx"])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # this would then become the integer input sequence to the transformer
    print("ready to feed into a Transformer!")
