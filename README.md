# Shifted VT-Codes 
A Shifted VT-Code is a P bounded single deletion/insertion correcting code.

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-blue?style=plastic)](./CONTRIBUTING.md)
![GitHub repo size](https://img.shields.io/github/repo-size/Guy-Shapira/Shifted-VT-codes?style=plastic)

The redundancy is ceil(log(P)) + 1, i.e. if a given word is of length n – (ceil(log(P)) + 1), then it can be encoded in n bits s.t. a single insertion/deletion can be corrected given its location up to P consecutive bits.

This repository implements the algorithms as described in paper: C. Schoeny, A. Wachter-Zeh, R. Gabrys, and E. Yaakobi, “Codes correcting a burst of deletions or insertions,” IEEE Transactions on Information Theory, vol. 63, no. 4, pp. 1971–1985, Apr. 2017.
# Code Constraint:
Shifted VT Codes are an extension to regular VT Codes.

In Shifted VT Codes,  all of codewords in the codespace are of length n, and the weighted sum of each codeword in the codespace is congruent to c (mod P) where c and P are code constants, moreover the parity of each codeword is d (another code constant). P is the size of the window in which we know an error had occurred.

# Usage:

## Encoding: 
```python
word = [0, 1, 1]
encoder = ShiftedVTCode.ShiftedVTCode(n=7, c=2, d=1, P=5)
codeword = encoder.encode(word)
print(codeword)  # output is '[0, 0, 0, 1, 0, 1, 1]'
```

Where n is a length of a codeword, c is the weighted sum, d is the parity and P is the maximum known distance of an error.

In the encoding we use ceil(log(P)) bits to correct the weighted sum of the given vector. We place those bits in the first ceil(log(P)) powers of two (indices 1, 2, 4, 8, ...). This enables us to represent any weighted sum that is lower than P, in this case, we set the value of those bits such that the weighted sum will be congruent to c (mod P).

The last redundancy bit is for correcting the parity of the encoded word. We place it in the P-th position in the vector (since in this index it won't affect the weighted sum).

## Decoding:
Decoding also works if no error has occurred, or if a single error (deletion/insertion) has occurred.

Continuation of the previous example:
```python
word = [0, 1, 1]
encoder = ShiftedVTCode.ShiftedVTCode(n=7, c=2, d=1, P=5)
code = encoder.encode(word)
print(codeword)  # output is '[0, 0, 0, 1, 0, 1, 1]'

# if no error has occured:
decoded = encoder.decode(codeword)
print(decoded)  # output is '[0, 1, 1]'

# if a single error (insertion or deletion) has occured at index 1
decoded = encoder.decode(codeword, u=0)
print(decoded)  # output is '[0, 1, 1]'
```

Correcting an insertion or deletion is identical in usage, thus only the index is passed, without the error type.

After correcting the error (if occurred) we remove all the redundancy bits, thus restoring the word to it's original form.

## Running the tests:
Unit tests are provided and are written using the `unittest` framework built into python.

Configure the framework to look for tests in files named `test_*`.
