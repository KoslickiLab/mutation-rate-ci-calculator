# mutation-rate-intervals

This software calculates a confidence interval for the mutation rate from
a set of observed containment indices under a simple nucleotide mutation process.
In this model, a sequence B evolves from a sequence A by independently mutating 
every nucleotide with probability `p`. We then observe the scaled containment indices
as a result of the mutation process. The software then generates a confidence interval
for the mutation rate `p`.

### Quick start

To compute a p confidence interval from an observed number of scaled containment indices:

```bash
p-from-scaled-containment.py -L 100K -k 21 -c 0.95 --sccon 0.10605
L       k       conf    Cks     CLow    CHigh   pLow    pHigh
100000  21      0.95    0.10605 0.10046 0.11191 0.09623 0.10655
```

#### Output
In the example above, the `L`, `k`, `conf`, and `Cks` values are the input number of k-mers, k-mer size, 
desired confidence level, and observed scaled MinHash value respectively. The `CLow`
and `CHigh` give the left and right 95% confidence interval around the true containment
value. The `pLow` and `pHigh` give a confidence interval that is at least as wide as the
95% confidence interval around the true mutation rate `p`. Note that average nucleotide
identity (ANI) is equal to `1-p`.

### How to choose L and other parameters
In reality, you may not know L. In such cases, we recommend that you estimate
it from what you know. For example, if what you know is that the number of
distinct (i.e. counting duplicates only once) k-mers in A is `nA` and in B is `nB`,
then you can set `L = (nA + nB) / 2`. You can also try setting `L = min(nA, nB)` or
`L = max(nA, nB)`.   

You may also want to get a confidence interval on `p` from the number
of mutated k-mers `N`, but you might only known the number of shared k-mers, i.e.
the number of k-mers in both A and B. If this number is `n`, then you can set
`N = L - n`.

Note that the programs consider `L` (uppercase) as the number of kmers in the
sequence, and `l` (lowercase) as the number of nucleotides, with `l = L + k-1`.

### Prerequisites

* python3
* scipy
* numpy

For computing confidence intervals, only scipy is required. An optional
module, mpmath, will be used if present (as described below).

Numpy is only used by the simulation programs.

Two addition packages are used if present: mpmath and mmh3.

mpmath is a multi-precision package, used here to avoid numerical problems that
can occur for very low mutation probabilities (e.g. 1e-8). If the module is not
present standard python floating-point is used instead.

mmh3 is a wrapper for MurmurHash3, used here for hashing kmers for bottom
sketches. If the module is not present the hashing options in
simulate_nucleotide_errors are not available.

### Usage Details

python p-from-scaled-containment.py

```bash
Compute confidence interval for the mutation rate p, given the observed number of mutated k-mers

optional arguments:
  -h, --help            show this help message and exit
  --sccon SCCON [SCCON ...]
                        observed MinHash Containment (input one or more values, separated by a space)
  --length LENGTH       number of nucleotides in the sequence
  -L NUM_UNIQUE_KMERS, --num-unique-kmers NUM_UNIQUE_KMERS
                        number of unique k-mers in the sequence
  --scaled SCALED       scaling factor of the sketch
  -k KSIZE, --ksize KSIZE
                        kmer size
  -c CONFIDENCE, --confidence CONFIDENCE
                        size of confidence interval, (value between 0 and 1)
  -s SEED, --seed SEED  random seed for simulations
  --debug               debug
  --debug_options [{nocache,nojmonotonicity,nsanity} [{nocache,nojmonotonicity,nsanity} ...]]
                        Specify one or more debugging options, separated by a space
```
