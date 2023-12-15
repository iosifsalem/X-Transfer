# implementation of xTransfer, crossPCN protocol

import numpy as np
import networkx as nx
import random
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import bisect 
import time
import matplotlib.pyplot as plt

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
    
    # unit test
    # txns = [Txn(client_name(0,0), client_name(1,0), G.edges[(client_name(0,0), 'hub0')]['capacity']*0.5), Txn(client_name(0,0), client_name(1,0), G.edges[(client_name(0,0), 'hub0')]['capacity'])]    
    
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
    
        # Create variables: binary for including a transaction or not
        x = m.addMVar(shape=len(txns), vtype=GRB.BINARY, name="x")
    
        # Set objective: maximize volume of successful transactions 
        obj = np.array([txn.amount for txn in txns])
        m.setObjective(obj @ x, GRB.MAXIMIZE)
    
        # Build (sparse) constraint and bound matrix
        # use matrix form: A*x <= b
        # i.e., each row A[i][*] of A multiplied by x is exactly the constraint A[i][*] * x <= b[i] 
        clients = [node for node in G.nodes if G.nodes[node]['label']=='client']
        
        # val[i] is the value of A in position (row[i], col[i])
        # the bound for row of A (i-th constraint) i is b[i] 
        val, row, col, b = [], [], [], []

        # optimization: the following for loops can be parallelized if needed
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

        # convert lists to np.array
        val = np.array(val)
        row = np.array(row)
        col = np.array(col)
        b = np.array(b)        

        # define A as a sparse matrix
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
    
    successful_txns = [txn for txn in txns if x.X[txns.index(txn)]]
    print(f'number of txns = {len(txns)}. successful txns = {len(successful_txns)}')

    success_volume = sum([txn.amount for txn in txns if x.X[txns.index(txn)]]) / sum([txn.amount for txn in txns])
    success_volume = np.floor(success_volume*100)/100  # two decimal precision 
    print(f'success volume = {success_volume}')
    return successful_txns, success_volume
    
def greedy_hub_flows(G, successful_txns):
    # computes flows among hubs that realize the successful transactions
    
    # compute in/out-flows
    flows = {node:0 for node in G.nodes if G.nodes[node]['label'] == 'hub'}
    
    for txn in successful_txns:
        sending_hub = hub_attached_to_client(txn.src)
        receiving_hub = hub_attached_to_client(txn.dst)
        flows[sending_hub] += txn.amount
        flows[receiving_hub] -= txn.amount

    print(flows)
    
    # sort flows
    hubs_with_outflow = []
    hubs_with_inflow = []
    for hub in flows:
        if flows[hub] >= 0:
            hubs_with_outflow.append([flows[hub], hub])
        else:
            hubs_with_inflow.append([flows[hub], hub])            
    
    # last element is the largest in absolute value 
    hubs_with_inflow.sort(reverse=True)
    hubs_with_outflow.sort()
    
    # print(f'in: {hubs_with_inflow}')
    # print(f'out: {hubs_with_outflow}')
        
    # satisfy demands (add remainder to sorted list)
    while hubs_with_inflow:
        (demand, rcv_hub) = hubs_with_inflow.pop()
        demand = abs(demand)
        
        while demand:
            # print(demand)
            # print(hubs_with_outflow)
            (supply, send_hub) = hubs_with_outflow.pop()
            if supply >= demand:
                G.add_edge(send_hub, rcv_hub, flow=demand)                
                supply -= demand
                demand = 0
                bisect.insort(hubs_with_outflow, [supply, send_hub])
                print(f"{G.edges[(send_hub, rcv_hub)]['flow']} from {send_hub} to {rcv_hub}")
            else:
                # check alg! probably insertion needed here too? 
                G.add_edge(send_hub, rcv_hub, flow=supply)
                print(f"{G.edges[(send_hub, rcv_hub)]['flow']} from {send_hub} to {rcv_hub}")                
                demand -= supply   

    # connect connected components 
    comps = [c for c in nx.strongly_connected_components(G)]
    for c in comps:
        print(c)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size = 500, with_labels=True)
    plt.show()
    
    return 0

def xtransfer(inpt):
    # X-Transfer computational part
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create PCNs using input parameters
    G, txns = createPCNsTxns(inpt)
    
    successfull_txns, success_volume = ILP(G, txns)
    
    hub_flows = greedy_hub_flows(G, successfull_txns)    
    
    #write output 
    
    return successfull_txns, success_volume, hub_flows
    
def graph1(inputs, duration):
    # plt.style.use('_mpl-gallery') 
    
    x = inputs
    y = duration
    
    # # plot
    # fig, ax = plt.subplots()
    
    # ax.plot(x, y)
    
    # # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    # #        ylim=(0, 8), yticks=np.arange(1, 8))
    
    # plt.show()
    
def graph2():
    x = 1
    
def graph3():
    x = 1

# input tuples in the form (#hubs or PCNs, #clients per hub, #txns/capacity percentage)
inputs = [(4,2,2)]
duration = []

for tpl in inputs:
    start = time.time()
    succesfull_txns, success_volume, hub_flows = xtransfer(tpl)
    end = time.time()
    duration.append(end - start)
    
# runtime
# graph1(inputs, duration)

# volume
graph2()

# sum of flows through hubs
graph3()