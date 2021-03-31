# mutation-rate-intervals

This software calculates a confidence interval for the mutation rate from
a set of observed containment indices under a simple nucleotide mutation process.
In this model, a sequence B evolves from a sequence A by independently mutating 
every nucleotide with probability p. We then observe the scaled containment indices
as a result of the mutation process. The software then generates a confidence interval
for the mutation rate p.

### Quick start

To compute a p confidence interval from an observed number of scaled containment indices:
```bash 
$ p-from-scaled-containment.py L=1M k=21 C=0.95 Cks=0.10605
L       k       conf    Cks     CLow    CHigh   pLow    pHigh
100000  21      0.95    0.10605 0.10046 0.11191 0.09623 0.10655
```

### How to choose L and other parameters
In reality, you may not know L. In such cases, we recommend that you estimate
it from what you know. For example, if what you know is that the number of
distinct (i.e. counting duplicates only once) k-mers in A is nA and in B is nB,
then you can set L = (nA + nB) / 2. You can also try setting L = min(nA, nB) or
L = max(nA, nB).   

You may also want to get a confidence interval on r<sub>1</sub> from the number
of mutated k-mers N, but you might only known the number of shared k-mers, i.e.
the number of k-mers in both A and B. If this number is n, then you can set
N = L - n.

Note that the programs consider L (uppercase) as the number of kmers in the
sequence, and l (lowercase) as the number of nucleotides, with l = L + k-1.

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

p-from-scaled-containment.py

```bash  
Compute confidence interval for the mutation rate r1, given the observed number
of mutated k-mers.

Compute confidence interval for the mutation rate p, given the observed number
of mutated k-mers.

usage: p-from-scaled-containment.py [options]
  --sccon=<list>              (Cks=) (cumulative) observed number of mutated
                              k-mers; <list> is a comma-separated list of
                              numbers
  --scale=<probability>       (s=) scaling factor of the hash
  
  --length=<N>                (l=) sequence length (number of NUCLEOTIDES in
                              the sequence)
                              (default is 1000 plus kmer size minus 1)
  L=<N>                       (L=) sequence length (number of KMERS in
                              the sequence)
                              (default is 1000000)
  --k=<N>                     (K=) kmer size
                              (default is 21)
  --confidence=<probability>  (C=) size of confidence interval
                              (default is 95%)
  --seed=<string>             random seed for simulations
```