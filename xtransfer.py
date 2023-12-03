# implementation of xTransfer, crossPCN protocol

import networkx as nx
import random

class Txn:
    def __init__(self, src, dst, amount):
        self.src = src
        self.dst = dst
        self.amount = amount
        
def createPCNsTxns(inpt):
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create a graph that denotes the connectivity among all PCNs
    G = nx.DiGraph()
    
    # we assume that all hubs can communicate with all other hubs
    # we will add the hub to hub links later, when we compute which ones are used
    G.add_nodes_from([f'hub{i}' for i in range(nPCNs)], label='hub')
    
    # TODO: fix capacity assignment
    for pcn in range(nPCNs):
        G.add_nodes_from([f'pcn{pcn}client{i}' for i in range(nClientsPerPCN)], label='client')
        G.add_edges_from([(f'hub{pcn}', f'pcn{pcn}client{i}') for i in range(nClientsPerPCN)], capacity=10)
        G.add_edges_from([(f'pcn{pcn}client{i}', f'hub{pcn}') for i in range(nClientsPerPCN)], capacity=10)

    # create transactions
    clients = [x for x in G.nodes if G.nodes[x]['label']=='client']
    
    txns = []
    for pcn in range(nPCNs):
        for client in range(nClientsPerPCN):
            # create random set of transactions summing up to capacity*txn_percentage
            amount_left = G.edges[(f'pcn{pcn}client{client}', f'hub{pcn}')]['capacity']
            while amount_left != 0:
                # select recepient
                recepients = list(clients).delete(client)
                
                # select amount
                txn_amount = random.randint()
                txns.append(())
            
    return G, txns

def ILP():
    x=1
    
def greedy():
    x=1

def xtransfer(inpt):
    # X-Transfer computational part
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create PCNs using input parameters
    pcns, txns = createPCNsTxns(inpt)
    
    #succesfull_txns = ILP(pcns, txns)
    
    #flows = greedy_flows(pcns, successfull_txns)
    # alg from Patcas paper might be faster (check!)
    
    #write output 
    
def graph1():
    x = 1
    
def graph2():
    x = 1
    
def graph3():
    x = 1

# input tuples in the form (#hubs or PCNs, #clients per hub, #txns/capacity percentage)
inputs = [(2,2,1)]

for tpl in inputs:
    xtransfer(tpl)
    
graph1()
graph2()
graph3()