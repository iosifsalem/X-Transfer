# X-Transfer
This repository relates to the paper "X-Transfer: Enabling and Optimizing Cross-PCN Transactions"
by 
Lukas Aumayr, Zeta Avarikioti, Iosif Salem, Stefan Schmid, and Michelle Yeo.


## Main files
* ```xtransfer.py```: implementation of X-transfer protocol's aggregation phase 
and its (proof of concept) performance evaluation.

* ```channels.csv```: channel info contained in a 
snapshot of the Lightning Network in August 2023.

## Input parameters for ```xtransfer.py```
Parametrization done by editing the following lines in xtransfer.py (sample values appear below):

```
nhubs = 5
nClientsPerPCN = 10000
# capacity_utilization is the ratio of sum of all transactions from a client 
# over the total client-to-hub channel capacity
# the ratio is the same for all clients and channels 
capacity_utilization = (0.5, 1, 2, 4, 8)
repetitions = 10  #number of times to compute each data point. Then take average.
plot_file_extension = '.pdf'
```

## Abstract
Blockchain interoperability solutions allow users to hold and transfer assets among different chains, and in so doing reap the benefits of each chain. To fully reap the benefits of multi-chain financial operations, it is paramount to support interoperability and cross-chain transactions also on Layer-2 networks, in particular payment channel networks (PCNs). Nevertheless, existing works on Layer-2 interoperability solutions still involve on-chain events, which limits their scalability and throughput. 

In this work, we present X-transfer, the first secure, scalable, and fully off-chain protocol that allows payments across different PCNs. We formalize and prove the security of X-transfer against rational adversaries with a game theoretic analysis. In order to boost efficiency and scalability, X-transfer also performs transaction aggregation to increase channel liquidity and transaction throughput while simultaneously minimizing payment routing fees. Our empirical evaluation of X-transfer shows that X-transfer achieves at least twice as much throughput compared to the baseline of no transaction aggregation, confirming X-transfer's efficiency.

**Keywords**: Payment channel networks, Layer-2, interoperability, optimization, transaction aggregation, cryptocurrencies
