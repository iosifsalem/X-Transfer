# X-Transfer
Implementation of a cross-PCN transaction aggregation protocol

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
