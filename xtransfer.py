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
        # TODO: parallelize the following for loop 
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

    # print(flows)
    
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
        
    # satisfy demands (add remainder to sorted list)
    while hubs_with_inflow:
        (demand, rcv_hub) = hubs_with_inflow.pop()
        demand = abs(demand)
        
        while demand:
            (supply, send_hub) = hubs_with_outflow.pop()
            if supply >= demand:
                G.add_edge(send_hub, rcv_hub, flow=demand)                
                supply -= demand
                demand = 0
                bisect.insort(hubs_with_outflow, [supply, send_hub])
                # print(f"{G.edges[(send_hub, rcv_hub)]['flow']} from {send_hub} to {rcv_hub}")
            else:
                # check alg! probably insertion needed here too? 
                G.add_edge(send_hub, rcv_hub, flow=supply)
                # print(f"{G.edges[(send_hub, rcv_hub)]['flow']} from {send_hub} to {rcv_hub}")                
                demand -= supply   

    # connect connected components 
    # heuristic: for every hub_w with 0 flow
    # remove an existing edge from hub_x to hub_y with flow f
    # add the edges hub_x --> hub_w --> hub_y, both with flow f
    initial_hub_to_hub_links = [[G.edges[(x,y)]['flow'], (x,y)] for (x,y) in G.edges if G.nodes[x]['label'] == 'hub' and G.nodes[y]['label'] == 'hub']
    initial_hub_to_hub_links.sort(reverse=True)
    for hub in flows:
        if flows[hub] == 0:
            hub_flow, (hub_from, hub_to) = initial_hub_to_hub_links.pop()
            G.remove_edge(hub_from, hub_to)
            G.add_edge(hub_from, hub, flow=hub_flow)
            G.add_edge(hub, hub_to, flow=hub_flow)            

    # heuristic, part2: connect non-zero-flow weakly connected components
    components = [c for c in nx.weakly_connected_components(G)]

    # connect every component A with the next one B
    for i in range(len(components)-1):
        hubsA = [node for node in components[i] if 'hub' in node]
        hubsB = [node for node in components[i+1] if 'hub' in node]
        hub_links_A = [[G.edges[(x,y)]['flow'], (x,y)] for x in hubsA for y in hubsA if (x,y) in G.edges]
        hub_links_B = [[G.edges[(x,y)]['flow'], (x,y)] for x in hubsB for y in hubsB if (x,y) in G.edges]
        hub_links_A.sort(reverse=True)
        hub_links_B.sort(reverse=True)        
        
        flowA, (s_A, d_A) = hub_links_A.pop()
        flowB, (s_B, d_B) = hub_links_B.pop()
        
        max_flow, (s_max, d_max) = max((flowA, (s_A, d_A)), (flowB, (s_B, d_B)))
        min_flow, (s_min, d_min) = min((flowA, (s_A, d_A)), (flowB, (s_B, d_B)))        
        
        G.remove_edge(s_min,d_min)
        G.add_edge(s_min, s_max, flow=min_flow)
        G.edges[(s_max,d_max)]['flow'] += min_flow
        G.add_edge(d_max, d_min, flow=min_flow)
    
    # print graph (if small)
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, node_size = 300, with_labels=True, node_shape='s')
    # plt.show()
    
    return [(G.edges[(x,y)]['flow'], (x,y)) for (x,y) in G.edges if G.nodes[x]['label'] == 'hub' and G.nodes[y]['label'] == 'hub']

def xtransfer(inpt):
    # X-Transfer computational part
    nPCNs, nClientsPerPCN, txn_percentage = inpt
    
    # create PCNs using input parameters
    G, txns = createPCNsTxns(inpt)
    
    # run ILP that computes the max (in volume) subset of feasible txns
    successfull_txns, success_volume = ILP(G, txns)
    
    # compute hub-to-hub flows
    hub_flows = greedy_hub_flows(G, successfull_txns) 
    sum_hub_flows = sum([tpl[0] for tpl in hub_flows])
    
    # for item in hub_flows:
    #     print(item)
    
    return success_volume, sum_hub_flows
    
def graph1(capacity_utilization, runtime):
    # plt.style.use('_mpl-gallery') 
    
    plt.plot(capacity_utilization, runtime, color='grey', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12) 
      
    # naming the x axis 
    plt.xlabel('(sum of txn amounts)/client-to-hub capacity') 
    # naming the y axis 
    plt.ylabel('runtime (s)') 
      
    # giving a title to my graph 
    plt.title('X-Transfer runtime ()') 
      
    # function to show the plot 
    plt.show()
    
def graph2():
    x = 1
    
def graph3():
    x = 1


# specify input parameters for generating all inputs 
nhubs = 5
nClientsPerPCN = 10
capacity_utilization = (0.5, 1, 2, 4, 8)
repetitions = 10

# input tuples in the form (#hubs or PCNs, #clients per hub, #txns/capacity percentage)
inputs = [(nhubs, nClientsPerPCN, util) for util in capacity_utilization]
outputs = {str(key):{'runtime':0, 'success_volume':0, 'sum_hub_flows':0} for key in capacity_utilization}

for tpl in inputs:
    key = str(tpl[2])
    
    # repeat each experiment {repetitions} number of times and take the average 
    for rep in range(repetitions):
        start = time.time()
        success_volume, sum_hub_flows = xtransfer(tpl)
        end = time.time()
    
        # record output
        outputs[key]['runtime'] += end - start
        outputs[key]['success_volume'] += success_volume
        outputs[key]['sum_hub_flows'] += sum_hub_flows
    
    # take average over number of repetitions
    outputs[key]['runtime'] /= repetitions
    outputs[key]['success_volume'] /= repetitions
    outputs[key]['sum_hub_flows'] /= repetitions
    
# runtime
graph1([tpl[2] for tpl in inputs], [outputs[key]['runtime'] for key in outputs])

# volume
graph2()

# sum of flows through hubs
graph3()