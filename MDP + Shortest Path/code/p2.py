xs = [-1, 0, 1, 2]
us = [-1, 1]

def next_state(x, u):
    return x * u + u**2 

for u in us:
    for x in xs:
        print(next_state(x, u))