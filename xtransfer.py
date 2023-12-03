# implementation of xTransfer, crossPCN protocol

import numpy as np
import networkx as nx
import random

class Txn:
    def __init__(self, src, dst, amount):
        self.src = src
        self.dst = dst
        self.amount = amount

def hub_name(pcn_id):
    return f'hub{pcn_id}'

def client_name(pcn_id, client_id):
    return f'pcn{pcn_id}client{client_id}'

# TODO: fix capacity assignment
def client_capacity_gen():
    return 10
        
def createPCNsTxns(inpt):
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create a graph that denotes the connectivity among all PCNs
    G = nx.DiGraph()
    
    # we assume that all hubs can communicate with all other hubs
    # we will add the hub to hub links later, when we compute which ones are used
    G.add_nodes_from([hub_name(i) for i in range(nPCNs)], label='hub')
    
    for pcn_id in range(nPCNs):
        G.add_nodes_from([client_name(pcn_id,i) for i in range(nClientsPerPCN)], label='client')
        hub = hub_name(pcn_id)
        G.add_edges_from([(hub, client_name(pcn_id, i)) for i in range(nClientsPerPCN)], capacity=client_capacity_gen())
        G.add_edges_from([(client_name(pcn_id, i), hub) for i in range(nClientsPerPCN)], capacity=client_capacity_gen())

    # create transactions
    clients = [x for x in G.nodes if G.nodes[x]['label'] == 'client']
    
    txns = []
    for pcn_id in range(nPCNs):
        hub = hub_name(pcn_id)
        for client_id in range(nClientsPerPCN):
            client = client_name(pcn_id, client_id)
            
            # create random set of transactions summing up to capacity*txn_percentage
            amount_left = np.floor(txn_percentage * G.edges[(client, hub)]['capacity'])
            while amount_left != 0:
                # select recepient
                dst = random.choice([node for node in clients if node != client])
                
                # select amount
                txn_amount = random.randint(1,amount_left)
                amount_left -= txn_amount
                txns.append(Txn(client, dst, txn_amount))           
    
    return G, txns

def ILP():
    x=1
    
def greedy():
    x=1

def xtransfer(inpt):
    # X-Transfer computational part
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create PCNs using input parameters
    G, txns = createPCNsTxns(inpt)
    
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