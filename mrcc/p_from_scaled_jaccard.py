#!/usr/bin/env python3

import sys
from math import ceil
import kmer_mutation_formulas_thm5 as thm5
import hypergeometric_slicer as hgslicer
from scipy.optimize import brentq, fsolve, newton
from scipy.stats import norm as scipy_norm
from numpy import sqrt
import argparse
from third_moment_calculator import *
from matplotlib import pyplot as plt
import third_moment_calculator as third

try:
    from mpmath import mp as mpmath,mpf
    mpmath.dps = 50
except ModuleNotFoundError:
    mpf = lambda v:float(v)


def probit(p):
    return scipy_norm.ppf(p)


def variance_scaled_jaccard(L, p, k, s):
    exp_n_mut = thm5.exp_n_mutated(L, k, p)
    exp_n_mut_squared = third.exp_n_mutated_squared(L, k, p)
    exp_n_mut_cubed = third.exp_n_mutated_cubed(L, k, p)
    bias_factor = 1 - (1 - s) ** ( int(L + exp_n_mut) )
    
    factor1 = (1-s)/(s * bias_factor**2)
    factor2 = (2 * L * exp_n_mut - 2 * exp_n_mut_squared) / (L ** 3 + 3*L*exp_n_mut_squared + 3*L*L*exp_n_mut + exp_n_mut_cubed)
    term1 = factor1 * factor2
    term2 = (L**2 - 2 * L * exp_n_mut + exp_n_mut_squared) / (L**2 + 2 * L * exp_n_mut + exp_n_mut_squared)
    term3 = ((L - exp_n_mut) / (L + exp_n_mut))**2
    
    return term1 + term2 - term3


def compute_confidence_interval_two_step(scaledJaccardsObserverved, L, k, confidence, s, debug=False):
    alpha = 1 - confidence
    z_alpha = probit(1-alpha/2)
    
    var = lambda N_mut : 2 * N_mut * (L - N_mut) * (1-s) / ( (L + N_mut)**3 * s )
    
    all_results = []
    for (JIx,Js) in enumerate(scaledJaccardsObserverved):
        print (Js)
        
        f1 = lambda N_mut: 1.0*(L - N_mut) / (L + N_mut) + z_alpha * sqrt( var(N_mut) )
        f2 = lambda N_mut: 1.0*(L - N_mut) / (L + N_mut) - z_alpha * sqrt( var(N_mut) )
        
        sol1 = brentq(f1, 1, L - 1)
        sol2 = brentq(f2, 1, L - 1)
        
        print (sol1, sol2)
        
        #values = [L,k,confidence,Js, sol2, sol1, 1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k)]
        #all_results.append(values)
    return all_results

def compute_confidence_interval_one_step(scaledJaccardsObserverved, L, k, confidence, s, debug=False):
    alpha = 1 - confidence
    z_alpha = probit(1-alpha/2)
    
    var_direct = lambda pest: variance_scaled_jaccard(L, pest, k, s)
        
    f1 = lambda pest: 2.0/(2- (1-pest)**k ) - 1 + z_alpha * sqrt(var_direct(pest)) - Js
    f2 = lambda pest: 2.0/(2- (1-pest)**k ) - 1 - z_alpha * sqrt(var_direct(pest)) - Js
    
    
    all_results = []
    for (JIx,Js) in enumerate(scaledJaccardsObserverved):
        #print (Js)
        if Js <= 0.0001:
            sol2 = sol1 = 1.0
        elif Js >= 0.9999:
            sol1 = sol2 = 0.0
        else:
            #sol1 = brentq(f1, max(1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k) - 0.05, 0.001), min(1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k) + 0.05,0.9999))
            #sol2 = brentq(f2, max(1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k) - 0.05, 0.001), min(1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k) + 0.05,0.9999))
            sol1 = brentq(f1, 0.0001, 0.9999)
            sol2 = brentq(f2, 0.0001, 0.9999)
        
        #print (var_direct(sol1))
        #print (var_direct(sol2))
        
        values = [L,k,confidence,Js, sol2, sol1, 1.0 - (2.0 - 2.0/(Js + 1))**(1.0/k)]
        all_results.append(values)
    return all_results

def main(args):
    global reportProgress,debug

    # parse the command line
    jaccardIndicesObserved=[]
    ntSequenceLength=None
    kmerSequenceLength=None
    jaccardIndicesObserved += list(map(parse_probability, args.scjaccard))
    if args.length:
        ntSequenceLength = int_with_unit(args.length)
    if args.num_unique_kmers:
        kmerSequenceLength = int_with_unit(args.num_unique_kmers)
    kmerSize = args.ksize
    scaleFactor = parse_probability(args.scaled)
    confidence = parse_probability(args.confidence)
    prngSeed = args.seed # not used anywhere?
    debug = []
    if args.debug_options:
       debug = ["debug"] + args.debug_options
    elif args.debug:
        debug = ["debug"]

    # check for necessary info
    if (kmerSequenceLength != None):
        ntSequenceLength = kmerSequenceLength + (kmerSize-1)
    elif (ntSequenceLength != None):
        kmerSequenceLength = ntSequenceLength - (kmerSize-1)

    # handle debugging options:
    if ("nocache" in debug):
        hgslicer.useCache = False
    if ("nojmonotonicity" in debug):
        hgslicer.doJMonotonicityCheck = False
    else:
        hgslicer.doJMonotonicityCheck = True
    if ("nsanity" in debug):
        hgslicer.doNLowSanityCheck  = True
        hgslicer.doNHighSanityCheck = True

    # compute the confidence interval(s)
    # todo: call here
    #conf_intervals = compute_confidence_intervals(scaledContainmentsObserverved,kmerSequenceLength,kmerSize,confidence,scaleFactor)
    #conf_intervals = compute_confidence_interval_test(scaledContainmentsObserverved,kmerSequenceLength,kmerSize,confidence,scaleFactor)
    #conf_intervals = compute_confidence_interval_one_step(scaledContainmentsObserverved,kmerSequenceLength,kmerSize,confidence,scaleFactor)

    #write results
    conf_intervals = compute_confidence_interval_one_step(jaccardIndicesObserved,kmerSequenceLength,kmerSize,confidence,scaleFactor)
    header = ["L","k","conf","Js","pLow","pHigh", "pointEst"]
    print("\t".join(header))
    for values in conf_intervals:
        print("\t".join(str(v)[:7] for v in values))
        

# parse_probability--
#    Parse a string as a probability

def parse_probability(s,strict=True):
    scale = 1.0
    if not isinstance(s, float):
        if (s.endswith("%")):
            scale = 0.01
            s = s[:-1]
        try:
            p = float(s)
        except:
            try:
                (numer,denom) = s.split("/",1)
                p = float(numer)/float(denom)
            except:
                raise ValueError
        p *= scale
    else:
        p=s
    if (strict) and (not 0.0 <= p <= 1.0):
        raise ValueError
    return p


# int_with_unit--
# Parse a string as an integer, allowing unit suffixes

def int_with_unit(s):
    if (s.upper().endswith("K")):
        multiplier = 1000
        s = s[:-1]
    elif (s.upper().endswith("M")):
        multiplier = 1000 * 1000
        s = s[:-1]
    elif (s.upper().endswith("G")):
        multiplier = 1000 * 1000 * 1000
        s = s[:-1]
    else:
        multiplier = 1

    try:               return          int(s)   * multiplier
    except ValueError: return int(ceil(float(s) * multiplier))



def cmdline(sys_args):
    "Command line entry point w/argparse action."
    p = argparse.ArgumentParser(description="Compute confidence interval for the mutation rate p, given the observed jaccard indices")
    p.add_argument("--scjaccard", nargs="+", help="observed MinHash Jaccard (input one or more values, separated by a space)", required=True) # at least one observation is required

    # add # nucleotides and the # of unique k-mers as a mutually exclusive group, with one (and only one) required
    seqlen_info = p.add_mutually_exclusive_group(required=True)
    seqlen_info.add_argument("--length", help="number of nucleotides in the sequence")
    seqlen_info.add_argument("-L", "--num-unique-kmers", help="number of unique k-mers in the sequence")
    # optional arguments
    p.add_argument("--scaled", help="scaling factor of the sketch", default=0.1)
    p.add_argument("-k", "--ksize", type=int, default=21, help="kmer size")
    p.add_argument("-c", "--confidence", default=0.95, help="size of confidence interval, (value between 0 and 1)") # type=float would constrain to float
    p.add_argument("-s", "--seed", type=int, help="random seed for simulations")
    p.add_argument("--debug", action="store_true", help="debug")
    p.add_argument("--debug_options", nargs="*", choices = ['nocache', 'nojmonotonicity', 'nsanity'], help="Specify one or more debugging options, separated by a space")
    args = p.parse_args()
    return main(args)

if __name__ == '__main__':
    cmdline(sys.argv)