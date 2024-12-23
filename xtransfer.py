# implementation of xTransfer, a privacy-preserving cross-PCN transaction aggregation protocol

import numpy as np
import networkx as nx
import random
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import bisect 
import time
import matplotlib.pyplot as plt
import json
import pandas as pd

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

def client_capacity_gen(array_length):
    # return an array of size array_length of randomly selected channel capacities 
    # from a Bitcoin Lightning Network snapshot (August 2023)  
    LN_channels = pd.read_csv("channels.csv")
    capacities = LN_channels.satoshis.values.tolist()  #satoshis is the dataframe column that gives the channnel capacity 
    return random.sample(capacities, array_length)
    # return list(np.random.randint(100, high=10000, size=array_length))
        
def createPCNsTxns(nPCNs, nClientsPerPCN, x, x_axis_legend, target_nTxns):    
    # create a graph that includes the links within all PCNs
    # we assume that all hubs can communicate with all other hubs
    # we will add the hub to hub links later, when we compute which ones are used
    G = nx.DiGraph()
    G.add_nodes_from([hub_name(i) for i in range(nPCNs)], label='hub')
    
    # client-hub/hub-clients channel capacity generation
    channel_capacities = client_capacity_gen(2*nPCNs*nClientsPerPCN)
    
    for pcn_id in range(nPCNs):
        G.add_nodes_from([client_name(pcn_id,i) for i in range(nClientsPerPCN)], label='client')
        hub = hub_name(pcn_id)
        G.add_edges_from([(hub, client_name(pcn_id, i)) for i in range(nClientsPerPCN)], capacity=channel_capacities.pop())
        G.add_edges_from([(client_name(pcn_id, i), hub) for i in range(nClientsPerPCN)], capacity=channel_capacities.pop())

    # create transactions
    # for each client, sum of txn amounts from that client = capacity*txn_percentage 
    clients = [x for x in G.nodes if G.nodes[x]['label'] == 'client']
        
    # unit test
    # txns = [Txn(client_name(0,0), client_name(1,0), G.edges[(client_name(0,0), 'hub0')]['capacity']*0.5), Txn(client_name(0,0), client_name(1,0), G.edges[(client_name(0,0), 'hub0')]['capacity'])]    
    txns = []

    if x_axis_legend == '(sum of txn amounts)/client-to-hub capacity':    
        for client in clients:
            for _ in range(int(target_nTxns/len(clients))):
                # create random set of transactions summing up to capacity*txn_percentage
                amount_left = int(x * G.edges[(client, hub_attached_to_client(client))]['capacity'])
                # select recepient
                dst = random.choice([node for node in clients if node != client])
                    
                # select amount
                if _ < int(target_nTxns/len(clients)) - 1:                    
                    txn_amount = random.randint(1,amount_left)
                else:
                    txn_amount = amount_left
                    
                amount_left -= txn_amount
                txns.append(Txn(client, dst, txn_amount))                      
                
    elif x_axis_legend == '#txns':                
        # txns between 5-4000 euro (12,637SATS - 10.11M SATS in Dec 2023)
        # todo (?): split in small/medium/high txn amounts (maybe according to local capacity) 
        
        lower_limit = 10_000  #satoshi
        upper_limit = 500_000  #satoshi 
        
        destination = lambda source : random.choice([node for node in clients if node != source])

        for i in range(x):
            src = clients[i%(len(clients))]
            capacity = G.edges[(src, hub_attached_to_client(src))]['capacity']
            if lower_limit < capacity:
                amount = random.randint(lower_limit, min(capacity, upper_limit))
            else:
                amount = random.randint(1000, capacity)
            txns.append(Txn(src, destination(src), amount))
                
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
        # m.Params.timelimit = 600  #set time limit (s)
    
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

        # for loop definition of var, row, col, b
        # val, row, col, b = [], [], [], []
        
        # for client in clients:
        #     hub = hub_attached_to_client(client)
        #     client_index = clients.index(client)
        #     for txn in txn_dict[client]['from']:
        #         val.append(txn.amount)
        #         row.append(client_index)
        #         col.append(txns.index(txn))
            
        #     for txn in txn_dict[client]['to']:
        #         val.append(-txn.amount)
        #         row.append(client_index)
        #         col.append(txns.index(txn))            
            
        #     b.append(G.edges[(client, hub)]['capacity'])

        # list comprehension 
        val = [txn.amount for client in clients for txn in txn_dict[client]['from']]        
        row = [clients.index(client) for client in clients for txn in txn_dict[client]['from']]
        col = [txns.index(txn) for client in clients for txn in txn_dict[client]['from']]

        val += [-txn.amount for client in clients for txn in txn_dict[client]['to']]        
        row += [clients.index(client) for client in clients for txn in txn_dict[client]['to']]
        col += [txns.index(txn) for client in clients for txn in txn_dict[client]['to']]
        
        b = [G.edges[(client, hub_attached_to_client(client))]['capacity'] for client in clients]

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
    
        # print(x.X)
        # print(f"Obj: {m.ObjVal:g}")
        
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    
    except AttributeError:
        print("Encountered an attribute error")
    
    successful_txns = [txn for txn in txns if x.X[txns.index(txn)]]
    print(f'number of txns = {len(txns)}. successful txns = {len(successful_txns)}')

    success_volume = sum([txn.amount for txn in txns if x.X[txns.index(txn)]]) / sum([txn.amount for txn in txns])
    success_volume = np.floor(success_volume*100)/100  # two decimal precision 
    print(f'success volume ratio = {success_volume}')
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
    # inflow hubs: first element largest in abs val, outflow hubs: last element largest in abs val
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

def xtransfer(G, txns):
    # X-Transfer computational part    
    # run ILP that computes the max (in volume) subset of feasible txns
    successfull_txns, success_volume = ILP(G, txns)
    
    # compute hub-to-hub flows
    hub_flows = greedy_hub_flows(G, successfull_txns) 
    sum_hub_flows = sum([tpl[0] for tpl in hub_flows])
    
    # for item in hub_flows:
    #     print(item)    
    
    return success_volume, sum_hub_flows

def no_aggregation(G, txns):
    # execute all txns without aggregation
    # output success ratio and sum of hub flows
    
    successful_txns_vol = 0
    total_vol = 0
    sum_hub_flows = 0
    
    # execute txns sequentially
    for txn in txns:
        cl_from, cl_to, amount = txn.src, txn.dst, txn.amount 
        total_vol += amount
        
        cl_from_hub = hub_attached_to_client(cl_from)
        cl_to_hub = hub_attached_to_client(cl_to)
        
        if G.edges[(cl_from, cl_from_hub)]['capacity'] >= amount and G.edges[(cl_to_hub, cl_to)]['capacity'] >= amount:
            # execute txn
            G.edges[(cl_from, cl_from_hub)]['capacity'] -= amount
            G.edges[(cl_from_hub, cl_from)]['capacity'] += amount            
            G.edges[(cl_to_hub, cl_to)]['capacity'] -= amount
            G.edges[(cl_to, cl_to_hub)]['capacity'] += amount            
            
            # add amount to metrics
            successful_txns_vol += amount
            sum_hub_flows += amount
            
    return successful_txns_vol/total_vol, sum_hub_flows

def graph_maker(x_axis, x_axis_legend, outputs, case, x_cases, nhubs, nClientsPerPCN, plot_file_extension):
    # plot runtime with increasing client-to-hub capacity utilization
    plt.cla()  #clear

    if case in {'success volume ratio', 'sum of hub flows'}:
        y = [outputs['X-Transfer'][x][case] for x in x_axis]
        plt.plot(x_axis, y, color='lightblue', linewidth = 3, 
             marker='o', markerfacecolor='blue', markersize=12) 

        z = [outputs['no aggregation'][x][case] for x in x_axis]
        plt.plot(x_axis, z, color='grey', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12) 
        plt.legend(['X-Transfer', 'no aggregation'])
    elif case == 'runtime (s)':
        y = [outputs['X-Transfer'][x]['mean runtime'] for x in x_axis]
        plt.plot(x_axis, y, color='lightblue', linewidth = 3, 
             marker='o', markerfacecolor='blue', markersize=12) 

        z = [outputs['X-Transfer'][x]['median runtime'] for x in x_axis]
        plt.plot(x_axis, z, color='lightgreen', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='green', markersize=12) 
        plt.legend(['mean', 'median'])
     
    plt.xlabel(x_axis_legend) 
    plt.ylabel(case) 
    plt.title(f'X-Transfer: {case}')    
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'outputs/H{nhubs}C{nClientsPerPCN}-{case} ({x_axis_legend.replace("/",":")}, {min(x_cases)}-{max(x_cases)}){plot_file_extension}')
    plt.show()

def run_scenario(nhubs, nClientsPerPCN, x_cases, x_axis_legend, repetitions, target_nTxns):
    # x_cases is either the capacity utilization values or the #txns values 
    
    outputs = {alg:{x:{'runtime (s)':[], 'success volume ratio':0, 'sum of hub flows':0} for x in x_cases} for alg in {'X-Transfer', 'no aggregation'}}
    
    #dict for plotting the txn amount distribution
    if x_axis_legend == '(sum of txn amounts)/client-to-hub capacity':
        txn_stats = {x:[0]*target_nTxns for x in x_cases}
    elif x_axis_legend == '#txns':
        txn_stats = {x:[0]*x for x in x_cases}   

    # run X-transfer and no aggregation algs over the input data and save results to output 
    for x in x_cases:    
        # repeat each experiment {repetitions} number of times and take the average 
        for _ in range(repetitions):
            # create PCNs using input parameters
            G, txns = createPCNsTxns(nhubs, nClientsPerPCN, x, x_axis_legend, target_nTxns)
            
            # txn amounts statistics 
            if x_axis_legend == '(sum of txn amounts)/client-to-hub capacity':
                list_size = target_nTxns
            elif x_axis_legend == '#txns':
                list_size = x
            txn_list = [(txns[i].amount, i) for i in range(list_size)]
            txn_list.sort()
            txn_stats[x] = [txn_stats[x][i] + txn_list[i][0]/repetitions for i in range(list_size)]
    
            # X-transfer
            start = time.time()
            X_transfer_success_volume, X_transfer_sum_hub_flows = xtransfer(G, txns)
            end = time.time()
        
            # record X-transfer output
            outputs['X-Transfer'][x]['runtime (s)'].append(end - start)
            outputs['X-Transfer'][x]['success volume ratio'] += X_transfer_success_volume
            outputs['X-Transfer'][x]['sum of hub flows'] += X_transfer_sum_hub_flows
            
            # no aggregation
            # X-Transfer computed the max feasible txns, but didn't apply them to the network (only added hub-to-hub links and their flows)
            # no aggregation ignores hub to hub links (assumes sufficient capacity) and applies all txns
            # thus no need to rest the client to hub channel capacities, as their initial values are intact
            no_agg_success_volume, no_agg_sum_hub_flows = no_aggregation(G, txns)
            outputs['no aggregation'][x]['success volume ratio'] += no_agg_success_volume
            outputs['no aggregation'][x]['sum of hub flows'] += no_agg_sum_hub_flows
        
        #runtime only relevant for X-Transfer
        outputs['X-Transfer'][x]['mean runtime'] = np.mean(outputs['X-Transfer'][x]['runtime (s)'])
        outputs['X-Transfer'][x]['median runtime'] = np.median(outputs['X-Transfer'][x]['runtime (s)'])        

        # take average over number of repetitions
        for alg in ['X-Transfer', 'no aggregation']:
            outputs[alg][x]['success volume ratio'] /= repetitions
            outputs[alg][x]['sum of hub flows'] /= repetitions
        
        # print txn distribution for the last x_case
        if x == x_cases[-1]:
            txn_amounts = [txn.amount for txn in txns]            
            outputs[f'{x} txns stats'] = {'min':min(txn_amounts), 'median':np.median(txn_amounts), 'mean':np.mean(txn_amounts), 'max':max(txn_amounts)}

            plt.cla()  #clear 
            plt.plot(range(list_size), txn_stats[x], color='darkgreen', linewidth = 2, 
              marker='o', markerfacecolor='green', markersize=5) 
            plt.xlabel('txn ranking by amount (lowest first)') 
            plt.ylabel('txn amount') 
            plt.title(f'txn distribution (avg over {repetitions} runs)')     
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'outputs/txn-distr-H{nhubs}C{nClientsPerPCN} ({x_axis_legend.replace("/",":")}, {min(x_cases)}-{max(x_cases)}){plot_file_extension}')
            plt.show()
            
            # save to output dict 
            outputs[f'{x} txn list, sorted and averaged ({repetitions} reps)'] = txn_stats[x]

    # save output
    filename = f'outputs/outputH{nhubs}C{nClientsPerPCN} ({x_axis_legend.replace("/",":")}, {min(x_cases)}-{max(x_cases)}).json'
    with open(filename, 'w') as handle:
        json.dump(outputs, handle)  

    ## uncomment for creating plots without rerunning the algs 
    # with open(filename, 'r') as handle:
    #     outputs = json.load(handle)
    
    # for key in {'X-Transfer', 'no aggregation'}:
    #     outputs[key] = {int(subkey):outputs[key][subkey] for subkey in outputs[key]}
    
    # create plots
    for case in ('runtime (s)', 'success volume ratio', 'sum of hub flows'):
        graph_maker(x_cases, x_axis_legend, outputs, case, x_cases, nhubs, nClientsPerPCN, plot_file_extension)    

## input parameters
# specify input parameters for generating all inputs 
nhubs = 5
# nClientsPerPCN = int(input("Insert #clients per PCN: "))
nClientsPerPCN = 1000

# scenario with increasing #txns
nTxns = (2000, 4000, 6000, 8000, 10000) 
# nTxns = (100, 200, 300)  # for debugging

# scenario with increasing capacity utilization: 
# capacity_utilization is the ratio of sum of all transactions from a client 
# over the total client-to-hub channel capacity
# the ratio is the same for all clients and channels 
capacity_utilization = (0.5, 1, 2, 4)
target_nTxns = 10*nClientsPerPCN

repetitions = 10  #number of times to compute each data point. Then take average.
plot_file_extension = '.pdf'

## scenarios 
x_axis_legend = '(sum of txn amounts)/client-to-hub capacity'
run_scenario(nhubs, nClientsPerPCN, capacity_utilization, x_axis_legend, repetitions, target_nTxns)

# x_axis_legend = '#txns'
# run_scenario(nhubs, nClientsPerPCN, nTxns, x_axis_legend, repetitions)
