import numpy as np
import string

wp = np.load('wp.npz')
wp = wp['text'].tobytes()
wp = wp.decode("utf-8")

def prob_matrix(s):
    probability = np.zeros(shape = (26,26))
    d = list(string.ascii_lowercase)
    for i in range(0,len(d)):
        for j in range(0,len(d)):
            count = 0
            for k in range(0,len(wp)):
                if(s[k:k+2] == d[i] + d[j]):
                    count += 1 
                probability[i][j] = count
    return probability

np.save('data.npz', prob_matrix(wp))



