# implementation of xTransfer, crossPCN protocol

def readInputs():
    return [(3, 3, 10)]
    
def xTransfer(nhubs, nclients, ntrans):
    return (1,2)

# input
# 
inputs = readInputs()

for triple in inputs:
    nhubs, nclients, ntrans = triple
    (runtime, cost) = xTransfer(nhubs, nclients, ntrans)
    
    