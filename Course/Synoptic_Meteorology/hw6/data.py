import numpy as np

with open ("output1.txt", "r") as myfile:
    #next(myfile)
    a = myfile.readline()

a = np.fromstring(a, sep=' ')

size = a.size
for i in range(size):
    if (abs(a[i]-a[0])-0.5)<0.05:
        print(i, a[0], a[i])

