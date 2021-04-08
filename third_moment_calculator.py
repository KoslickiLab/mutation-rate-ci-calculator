from scipy.special import hyp2f1
from mutation_model_simulator import MutationModel

def third_moment_nmut(L,k,p):
    t1 = (L * (-2 + 3*L) * p**2 + 3 * (1 - p)**(2*k) * (2 + (-1 + k - L) * p * (2 + k * p - L * p)) - (1 - p)**k * (6 + p * (-6 + L * (-6 + p + 6 * L * p))))/(p**2)
    t2 = (-2 + 2 * k - L) * (-1 + 2 * k - L) * (2 * k - L) * (-1 + (1 - p)**k)**3
    t3 = (1/(p**3))*(-6 * (-1 + k)**2 * (k - L) * p**3 + 6 * (1 - p)**(3 * k) * (2 + (-2 + 2 * k - L) * p) + (1 - p)**(2 * k) * (-12 + 6 * (2 * k + L) * p + 6 * (4 * k**2 + 2 * (1 + L) - 3 * k * (2 + L)) * p**2 - (-1 + k) * k * (-2 + 4 * k - 3 * L) * p**3) + 6 * (-1 + k) * (1 - p)**k * p * (-2 + p * (2 - k + 2 * L + (k * (-2 + 3 * k - 3 * L) + L) * p)))
    t4 = 6 * (-1 + (1 - p)**k) * ((k + k**2 - 2 * k * L + (-1 + L) * L) * (-1 + 2 * (1 - p)**k) * hyp2f1(1, 2 + k - L, k - L, 1) + (k + k**2 - 2 * k * L + (-1 + L) * L) * (1 - p)**k * (-1 + p) * hyp2f1(1, 2 + k - L, k - L, 1 - p) - (-2 * k + 4 * k**2 + L - 4 * k * L + L**2) * ((-1 + 2 * (1 - p)**k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1)- (-1 + p)**(2 * k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1 - p)))
    return t1+t2+t3+t4    
    

# testing code
if __name__=='__main__':
    kmerSequenceLength = 100
    kmerSize = 21
    pSubstitution = 0.05
    numIterations = 1000000
    
    nMutatedList = []
    mutationModel = MutationModel(kmerSequenceLength+kmerSize-1,kmerSize,pSubstitution)
    for trialNumber in range(numIterations):
        mutationModel.generate()
        nErrors, nMutated = mutationModel.count()
        nMutatedList.append(nMutated)
        
    nMutCubeSum = sum([x**3 for x in nMutatedList])
    nMutCubeEstimated = 1.0*nMutCubeSum/numIterations
    print('Estimated from simulations: \t' + str(nMutCubeEstimated))
    
    m = third_moment_nmut(kmerSequenceLength, kmerSize, pSubstitution)
    print('From formula: \t\t\t' + str(m))
    
    print('% difference: \t\t\t' + str(100*abs(nMutCubeEstimated-m)/nMutCubeEstimated))