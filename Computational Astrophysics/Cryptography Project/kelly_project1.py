# Collaboration with Huy Tran, Ben Sherwin, Marcus Mynatt, Darius Scott, and Brian Phillips

import numpy as np
import string
import random
import matplotlib.pyplot as plt

probability = np.load('data.npz.npy')
probability = probability/(np.sum(probability))

alp = list(string.ascii_lowercase)
x = list(string.ascii_lowercase)

freq = np.load('data.npz.npy')
logspace = np.log(freq+1)

phrase = 'zywdynfzmbboanxjrxiaimbbxpgaxwiyrymbpgoyxal'


## Functions ###


def key():
    random.shuffle(x)
    key = {alp[i]:x[i] for i in range(len(alp))}
    return key  


def decode(key,s):
    jumbled = ''
    for i in range(len(s)):
        convert = key[phrase[i]]
        jumbled += convert
    return jumbled


def swap(key):
    rand1 = np.random.randint(0,25)
    rand2 = np.random.randint(0,25)
    key2 = key.copy()
    s = key2[alp[rand1]]
    key2[alp[rand1]] = key2[alp[rand2]]
    key2[alp[rand2]] = s
    return key2


def scoring_key(distribution):
    pairs = []
    for i in range(len(alp)):
        for k in range(len(alp)):
            pairs.append(alp[i]+alp[k])
    scoring_key = {pairs[i]:np.ravel(distribution)[i] for i in range(len(pairs))}
    return scoring_key


def scoring(phrase, scoring_key):
    counter = 0
    for i in range(len(phrase)-1):
        counter += scoring_key[phrase[i] + phrase[i+1]]
    return counter


def correctness_percent(decoded):
    correct_phrase = 'jackandjillwentupthehilltofetchapaleofwater'
    correct_counter = 0
    for i in range(len(decoded)):
        if decoded[i] == correct_phrase[i]:
            correct_counter += 1
    return correct_counter*100/len(decoded)


def annealing(score1, score2, T):
    return (np.random.uniform() < np.exp(-np.abs((score2-score1))/T))


### Main Code ###


def MCMC_NoAnnealing(N_SIMS):
    scores = []
    cipher = key()
    phrase = 'zywdynfzmbboanxjrxiaimbbxpgaxwiyrymbpgoyxal'
    scoring_key1 = scoring_key(logspace)
    for i in range(N_SIMS): 
        P = scoring(decode(cipher, phrase), scoring_key1)
        swapped_cipher = swap(cipher)
        P2 = scoring(decode(swapped_cipher, phrase), scoring_key1)
        if P2 > P:
            # keep swap
            cipher = swapped_cipher
            scores.append(P2)
        else:
            scores.append(P)
    decoded = decode(cipher, phrase)
    return decoded, scores


def MCMC_Annealing(Tmax, Tmin, tau):
    scores = []
    cipher = key()
    phrase = 'zywdynfzmbboanxjrxiaimbbxpgaxwiyrymbpgoyxal'
    scoringkey1 = scoring_key(logspace)
    t = 0
    T = Tmax
    while T>Tmin:
        t += 1
        T = Tmax * np.exp(-t/tau)
        P = scoring(decode(cipher, phrase), scoringkey1)
        swapped_cipher = swap(cipher)
        P2 = scoring(decode(swapped_cipher, phrase), scoringkey1)
        if P2>P:
            cipher = swapped_cipher
            scores.append(P2)
        elif (annealing(np.log(P), np.log(P2), T)):
            cipher = swapped_cipher
            scores.append(P2)
        else:
            scores.append(P)
    decoded = decode(cipher, phrase)
    return [decode(cipher, phrase), scores]


decoded_no_annealing, scores_no_annealing = MCMC_NoAnnealing(1000000)
decoded_annealing, scores_annealing = MCMC_Annealing(10000,1e-4,1e3)

percent1 = correctness_percent(decoded_no_annealing)
percent2 = correctness_percent(decoded_annealing)

print(percent1,'no_annealing')
print(percent2,'annealing')


### Plotting Archive ###


plt.plot(np.arange(len(scores_no_annealing)), scores_no_annealing)
plt.ylabel('Scores')
plt.xlabel('Number of Trials')
plt.title('No Annealing Scores for 1000000 Trials')
plt.savefig('No_Annealing_1000000.png')

plt.close()
plt.plot(np.arange(len(scores_annealing)), scores_annealing)
plt.ylabel('Scores')
plt.xlabel('Number of Trials')
plt.title('Annealing Scores for Tmax = 10000, Tmin = 1e-4, Tau = 1e3')
plt.savefig('Annealing_10000_1e-4_1e3.png')
