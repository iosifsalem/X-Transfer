# implementation of xTransfer, crossPCN protocol

import numpy as np
import networkx as nx
import random
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

class Txn:
    def __init__(self, src, dst, amount):
        self.src = src
        self.dst = dst
        self.amount = amount

def hub_name(pcn_id):
    return f'hub{pcn_id}'

def client_name(pcn_id, client_id):
    return f'pcn{pcn_id}client{client_id}'

def hub_attached_to_client(client_name):
    return f"hub{client_name.split('client')[0][3:]}"

# TODO: fix capacity assignment
def client_capacity_gen():
    return 10
        
def createPCNsTxns(inpt):
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create a graph that includes the links within all PCNs
    # we assume that all hubs can communicate with all other hubs
    # we will add the hub to hub links later, when we compute which ones are used
    G = nx.DiGraph()
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

def ILP(G, txns):
    # we run an ILP to select the max feasible transactions in volume
    # we contract the hub nodes to one central node that forms a star with all clients
    # we use Gurobi to compute the ILP solution
    
    # create dictionary to distinguish transactions to/from a specific client
    txn_dict = {node:{'from':[], 'to':[]} for node in G.nodes if G.nodes[node]['label'] == 'client'}
    for txn in txns:
        txn_dict[txn.src]['from'].append(txn)
        txn_dict[txn.dst]['to'].append(txn)
    
    try:
        # Create a new model
        m = gp.Model("transaction-selection")
    
        # Create variables
        x = m.addMVar(shape=len(txns), vtype=GRB.BINARY, name="x")
    
        # Set objective
        obj = np.array([txn.amount for txn in txns])
        m.setObjective(obj @ x, GRB.MAXIMIZE)
    
        # Build (sparse) constraint and bound matrix
        # use matrix form: A*x <= b
        # val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
        # row = np.array([0, 0, 0, 1, 1])
        # col = np.array([0, 1, 2, 0, 1])
        clients = [node for node in G.nodes if G.nodes[node]['label']=='client']
        val, row, col, b = [], [], [], []

        for client in clients:
            hub = hub_attached_to_client(client)
            client_index = clients.index(client)
            for txn in txn_dict[client]['from']:
                val.append(txn.amount)
                row.append(client_index)
                col.append(txns.index(txn))
            
            for txn in txn_dict[client]['to']:
                val.append(-txn.amount)
                row.append(client_index)
                col.append(txns.index(txn))            
            
            b.append(G.edges[(client, hub)]['capacity'])
            
        for lst in [val, row, col, b]:
            lst = np.array(lst)
        
        A = sp.csr_matrix((val, (row, col)), shape=(len(clients), len(txns)))
        
        # Add constraints
        m.addConstr(A @ x <= b, name="c")
    
        # Optimize model
        m.optimize()
    
        print(x.X)
        print(f"Obj: {m.ObjVal:g}")
    
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    
    except AttributeError:
        print("Encountered an attribute error")
    
    successful_txns = txns
    return successful_txns
    
def greedy():
    x=1

def xtransfer(inpt):
    # X-Transfer computational part
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create PCNs using input parameters
    G, txns = createPCNsTxns(inpt)
    
    succesfull_txns = ILP(G, txns)
    
    #flows = greedy_flows(pcns, successfull_txns)
    # alg from Patcas paper might be faster (check!)
    
    #write output 
    
    return txns
    
def graph1():
    x = 1
    
def graph2():
    x = 1
    
def graph3():
    x = 1

# input tuples in the form (#hubs or PCNs, #clients per hub, #txns/capacity percentage)
inputs = [(2,2,1)]

for tpl in inputs:
    txns = xtransfer(tpl)
    
graph1()
graph2()
graph3()