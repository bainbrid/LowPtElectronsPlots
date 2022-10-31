import numpy as np

def interpolate(x,values):
    if 'int' in type(x).__name__: return values[x]
    else: # interpolate
        dx = x - int(x)
        if dx == 0.:
            return values[int(x)]
        else:
            y = values[int(x)]
            dy = values[int(x)+1] - y
            return y + dy*dx

# poisson intervals (68% CL)
def poisson(x):
    lower = None
    upper = None
    if x > 19. :
        lower = np.sqrt(x)
        upper = np.sqrt(x)
        return lower,upper
    lowers = [
        0.,0.,0.827246,1.29181,1.6327,1.91434,2.15969,2.37993,2.58147,2.76839, # 0->9
        2.94346,3.10869,3.26558,3.41527,3.55866,3.6965,3.82938,3.9578,4.08218,4.20289, # 10-19
        4.32022 # 20
        ]
    lower = interpolate(x,lowers)
    uppers = [
        1.84102,1.84102,2.29953,2.63786,2.91819,3.16275,3.38247,3.58364,3.77028,3.94514, # 0->9
        4.1102,4.26695,4.41652,4.55982,4.69757,4.83038,4.95874,5.08307,5.20372,5.32101, # 10-19
        5.4352 # 20
        ]
    upper = interpolate(x,uppers)
    return lower,upper

if __name__ == "__main__":

    ints = range(0,10)
    lowers,uppers = list(zip(*[ poisson(i) for i in ints ]))
    print("Ints:")
    for i,l,u in zip(ints,lowers,uppers):
        print(f"central: {i:5.2f} lower: {l:5.2f}, upper: {u:5.2f}")

    floats = np.arange(0.,20., 0.5)
    lowers,uppers = list(zip(*[ poisson(f) for f in floats ]))
    print("Floats:")
    for f,l,u in zip(floats,lowers,uppers):
        print(f"central: {f:5.2f} lower: {l:5.2f}, upper: {u:5.2f}")
        
