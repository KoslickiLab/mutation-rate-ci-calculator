from scipy.special import hyp2f1
from mutation_model_simulator import MutationModel
from kmer_mutation_formulas_thm5 import exp_n_mutated, var_n_mutated
from matplotlib import pyplot as plt
import kmer_mutation_formulas_thm5 as thm5

def var_c_scaled_first_order_taylor(L,k,p,s):
    q = 1 - (1 - p) ** k
    m = L - L*q
    n = L*q
    #print(n)
    #print(m)
    return 1.0 * m * n * (1-s) / (L**3 * s)

def var_test(L,k,p,s,confidence):
    print ('test')
    print (thm5.var_n_mutated(L, k, p)/L**2)

def var_c_scaled_one_step(L,k,p,s,confidence):
    bias_factor = 1 - (1 - s) ** L
    var_multiplier = (1-s)**2 / (L**6 * s**2 * bias_factor**2)
    var_inner = lambda pest: L**2 * thm5.var_n_mutated(L, k, pest) + var_n_mutated_squared(L, k, pest) - 2*L* ( exp_n_mutated_cubed(L, k, pest) - exp_n_mutated(L,k,pest) * exp_n_mutated_squared(L, k, pest) )
    var_direct = lambda pest: var_multiplier * var_inner(pest)
    print (var_inner(p))
    print (var_multiplier)
    return var_direct(p)

def third_moment_nmut_exact(L,k,p):
    t1 = (L * (-2 + 3*L) * p**2 + 3 * (1 - p)**(2*k) * (2 + (-1 + k - L) * p * (2 + k * p - L * p)) - (1 - p)**k * (6 + p * (-6 + L * (-6 + p + 6 * L * p))))/(p**2)
    t2 = (-2 + 2 * k - L) * (-1 + 2 * k - L) * (2 * k - L) * (-1 + (1 - p)**k)**3
    t3 = (1/(p**3))*(-6 * (-1 + k)**2 * (k - L) * p**3 + 6 * (1 - p)**(3 * k) * (2 + (-2 + 2 * k - L) * p) + (1 - p)**(2 * k) * (-12 + 6 * (2 * k + L) * p + 6 * (4 * k**2 + 2 * (1 + L) - 3 * k * (2 + L)) * p**2 - (-1 + k) * k * (-2 + 4 * k - 3 * L) * p**3) + 6 * (-1 + k) * (1 - p)**k * p * (-2 + p * (2 - k + 2 * L + (k * (-2 + 3 * k - 3 * L) + L) * p)))
    t4 = 6 * (-1 + (1 - p)**k) * ((k + k**2 - 2 * k * L + (-1 + L) * L) * (-1 + 2 * (1 - p)**k) * hyp2f1(1, 2 + k - L, k - L, 1) + (k + k**2 - 2 * k * L + (-1 + L) * L) * (1 - p)**k * (-1 + p) * hyp2f1(1, 2 + k - L, k - L, 1 - p) - (-2 * k + 4 * k**2 + L - 4 * k * L + L**2) * ((-1 + 2 * (1 - p)**k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1)- (-1 + p)**(2 * k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1 - p)))
    return t1+t2+t3+t4    
    
def exp_n_mutated_squared(L, k, p):
    return var_n_mutated(L, k, p) + exp_n_mutated(L, k, p) ** 2

def exp_n_mutated_cubed(L, k, p):
    return third_moment_nmut_exact(L, k, p)

def exp_n_mutated_to_the_fourth_power(L, k, p):
    return fourth_moment_using_normal(L, k, p)

def var_n_mutated_squared(L, k, p):
    return exp_n_mutated_to_the_fourth_power(L, k, p) - exp_n_mutated_squared(L, k, p) ** 2

def third_moment_nmut_using_normal(L,k,p):
    mu = exp_n_mutated(L, k, p)
    var = var_n_mutated(L, k, p)
    third_moment = mu**3 + 3 * mu * var
    return third_moment

def fourth_moment_using_normal(L,k,p):
    mu = exp_n_mutated(L, k, p)
    var = var_n_mutated(L, k, p)
    fourth_moment = mu**4 + 6 * mu**2 * var + 3 * var**2
    return fourth_moment

def generate_third_moment_stats(L_low,L_high,dL,k,p,perform_simulation=True,num_itertions=1000):
    stats = []
    for L in range(L_low,L_high,dL):
        if perform_simulation:
            mutation_model = MutationModel(L+k-1,k,p)
            n_mut_list = []
            for trial_number in range(num_itertions):
                mutation_model.generate()
                n_errors, n_mutated = mutation_model.count()
                n_mut_list.append(n_mutated)
            n_mut_cube_sim = 1.0*sum([x**3 for x in n_mut_list])/num_itertions
        n_mut_cube_normal = third_moment_nmut_using_normal(L,k,p)
        n_mut_cube_exact = third_moment_nmut_exact(L,k,p)
        if perform_simulation:
            stats.append((L,n_mut_cube_sim, n_mut_cube_exact, n_mut_cube_normal))
        else:
            stats.append((L,n_mut_cube_exact, n_mut_cube_normal))
    return stats

def generate_fourth_moment_stats(L_low,L_high,dL,k,p,num_itertions=1000):
    stats = []
    L_ranges = []
    L = L_low
    while L <= L_high:
        L_ranges.append(L)
        L = L * dL
    for L in L_ranges:
        print(L)
        mutation_model = MutationModel(L+k-1,k,p)
        n_mut_list = []
        for trial_number in range(num_itertions):
            mutation_model.generate()
            n_errors, n_mutated = mutation_model.count()
            n_mut_list.append(n_mutated)
        n_mut_fourth_sim = 1.0*sum([x**4 for x in n_mut_list])/num_itertions
        n_mut_fourth_normal = fourth_moment_using_normal(L,k,p)
        stats.append((L,n_mut_fourth_sim, n_mut_fourth_normal))
    return stats

# testing code
if __name__=='__main__':
    L = 1000
    k = 21
    conf = 0.95
    scale_factor = 0.1
    mutation_rate = 0.03
    
    print(var_c_scaled_one_step(L, k, mutation_rate, scale_factor, conf))
    print(var_test(L, k, mutation_rate, scale_factor, conf))
    print(var_c_scaled_first_order_taylor(L, k, mutation_rate, scale_factor))
    
    
    '''
    kmerSequenceLengthLow = 100
    kmerSequenceLengthHigh = 1000000
    stepSize = 5
    kmerSize = 21
    pSubstitution = 0.05
    numIterations = 10000
    
    stats = generate_third_moment_stats(kmerSequenceLengthLow,kmerSequenceLengthHigh,
                                        stepSize,kmerSize,pSubstitution,False)
    plt.plot([x[0] for x in stats], [x[1] for x in stats])
    plt.plot([x[0] for x in stats], [x[2] for x in stats])
    plt.show()
    
    plt.plot([x[0] for x in stats], [ 100*abs(x[1]-x[2])/x[1] for x in stats ])
    plt.show()
    
    stats_fourth_moment = generate_fourth_moment_stats(kmerSequenceLengthLow,kmerSequenceLengthHigh,
                                        stepSize,kmerSize,pSubstitution, numIterations)
    plt.plot([x[0] for x in stats_fourth_moment], [ 100*abs(x[1]-x[2])/x[1] for x in stats_fourth_moment ])
    plt.show()
    '''
    