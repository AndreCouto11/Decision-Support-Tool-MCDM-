from gurobipy import Model, GRB
from math import factorial
import random
import signal
import math
import pandas as pd
import itertools
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt


value_changes = {
    1: [1],
    2: [3, 4, 5, 6],
    3: [2, 4, 5, 6],
    4: [2, 3, 5, 6],
    5: [ 3, 4, 6, 7],
    6: [ 4, 5, 7, 8],
    7: [ 5, 6, 8, 9],
    8: [ 6, 7, 9],
    9: [  7, 8]
}


def calculate_weights_and_v( num_criteria, preferences):
    # Create the pairwise comparison matrix
    matrix = np.ones((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(i+1, num_criteria):
            if (i, j) in preferences:
                matrix[i, j] = preferences[(i, j)]
                matrix[j, i] = 1 / preferences[(i, j)]
            else:
                matrix[j, i] = preferences[(j, i)]
                matrix[i, j] = 1 / preferences[(j, i)]
    
    #print(matrix)

    # Calculate the eigenvalues and corresponding eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Calculate the largest eigenvalue and corresponding eigenvector
    max_eigenvalue_index = np.argmax(eigenvalues)
    principal_eigenvector = np.real(eigenvectors[:, max_eigenvalue_index])

    # Normalize the principal eigenvector to get the weights
    weights = principal_eigenvector / np.sum(principal_eigenvector)

    #print(weights)

    # Calculate the eigenvalues and eigenvectors of the transpose of the matrix
    eigenvalues_T, eigenvectors_T = np.linalg.eig(matrix.T)

    # Select the eigenvector corresponding to the largest eigenvalue of A^T
    max_eigenvalue_index_T = np.argmax(eigenvalues_T)
    principal_eigenvector_T = np.real(eigenvectors_T[:, max_eigenvalue_index_T])

    # Normalize this eigenvector such that its dot product with w equals 1
    v = principal_eigenvector_T / np.dot(principal_eigenvector_T, weights)
    v= v/np.sum(v)
    #print(v)
    return weights, v, matrix


def calculate_consistency_ratio(matrix, num_criteria):
    # Calculate the largest eigenvalue (lambda_max)
    eigenvalues, _ = np.linalg.eig(matrix)
    lambda_max = np.real(np.max(eigenvalues))

    # Calculate the consistency index (CI)
    CI = (lambda_max - num_criteria) / (num_criteria - 1)

    # Random Consistency Index (RI) values for n=1 to 10
    RI_values = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    # Get the RI value for the current number of criteria
    RI = RI_values[num_criteria - 1]

    # Calculate the consistency ratio (CR)
    CR = CI / RI

    return CR



def adjust_preferences(num_criteria, preferences, weights, matrix):
    initial_preferences = preferences.copy()
    iteration = 0
    #T = 0.9  # Initial temperature
    #T_min = 0.000001  # Minimum temperature
    #alpha = 0.8  # Cooling rate
    consistency_ratios = []
    matrices = []
    weights_list = []
    matrices.append(matrix)
    weights_list.append(weights)
    consistency_ratio = calculate_consistency_ratio(matrix, num_criteria)
    print("initial consistency: ", consistency_ratio)
    #print(weights)
    #print(matrix)
    consistency_ratios.append(consistency_ratio)
    weights, v, _ = calculate_weights_and_v(num_criteria, preferences)
    #print("v= ",v)

    values = {}
    for i, j in preferences.keys():

                #print(i,j)
        if j > i:
                vi = v[i]
                vj = v[j]
                aji = matrix[j, i]
                wj = weights[j]
                wi = weights[i]
                value = abs(vi*wj - (aji**2)*vj*wi)
                #print(value)
                values[(i, j)] = value
        
                #if (i, j) == (1, 3):
                    #print(vi,vj,aji,wj,wi)
                    #print(f"The value for preference (3,8) is: {value}")
            # Sort the cells by their values in descending order
                sorted_cells = sorted(values, key=values.get, reverse=True)
                #print(sorted_cells)
        else:
            vi = v[j]
            vj = v[i]
            aji = matrix[i, j]
            wj = weights[i]
            wi = weights[j]
            value = abs(vi*wj - (aji**2)*vj*wi)
            #print(value)
            values[(j, i)] = value
            sorted_cells = sorted(values, key=values.get, reverse=True)
            #print(sorted_cells)
            #print(sorted_cells)
    if consistency_ratio < 0.1:
        weights= weights
        matrix = matrix
        weights_list.append(weights)
    else:
        while iteration <= 100:
            iteration += 1
            #print(iteration)
            #print(iteration)
            # Calculate the values of vi*wj-(aji)^2*vj*wi for all cells
            
            for i, j in sorted_cells:
                inverse = False
                if (i, j) in preferences:
                    old_value = preferences[(i, j)]
                else:
                    old_value = preferences[(j, i)]
                    inverse = True
                best_value = old_value
                best_ratio = consistency_ratio
                possible_values = value_changes[initial_preferences[(i,j)]] if not inverse else value_changes[initial_preferences[(j,i)]]
                if old_value < 1:
                    possible_values = [1 / k for k in possible_values]
                for new_value in possible_values:
                    if not inverse:
                        preferences[(i, j)] = new_value
                    else:
                        preferences[(j, i)] = new_value
                    new_weights, new_v, new_matrix = calculate_weights_and_v(num_criteria, preferences)
                    new_consistency_ratio = calculate_consistency_ratio(new_matrix, num_criteria)
                    if new_consistency_ratio < best_ratio:
                        best_ratio = new_consistency_ratio
                        best_value = new_value
                    if best_ratio < 0.1:
                        break
                    #print(new_value, new_consistency_ratio)
                if inverse:
                    preferences[(j, i)] = best_value
                else:
                    preferences[(i, j)] = best_value
                weights, _, matrix = calculate_weights_and_v(num_criteria, preferences)
                consistency_ratio = calculate_consistency_ratio(matrix, num_criteria)
                #print("altera ",matrix)
                #print("altera correspondente ", consistency_ratio)
                if consistency_ratio < 0.1:
                 break
            consistency_ratios.append(consistency_ratio)
            matrices.append(matrix)
            weights_list.append(weights)
            if consistency_ratio < 0.1:
                break

        #print(sorted_cells) 
             
        initial_matrix = np.array(matrices[0])
        print("initial ", initial_matrix)
        final_matrix = np.array(matrices[-1])
        print("final", final_matrix)
        print(consistency_ratio)
        delta_max = np.max(np.abs(final_matrix - initial_matrix))
        sigma = np.sqrt(np.sum((final_matrix - initial_matrix)**2)) / (num_criteria)
        # Count the number of changed preferences
        num_changed_preferences = np.sum(initial_matrix != final_matrix) / 2

        print("Delta Max:", delta_max)
        print("Sigma:", sigma)
        print("Num changed preferences", num_changed_preferences)
    return weights_list, matrices, consistency_ratios




def calculate_feedback_arcset(graph, order):
    feedback_edges = []
    for edge, weight in graph.items():
        if order.index(edge[0]) > order.index(edge[1]):
            feedback_edges.append((edge, weight))
    return feedback_edges

def minimum_feedback_arcset(graph):
    # Add reverse edges for edges with weight 1
    for edge, weight in list(graph.items()):
        if weight == 1.0:
            graph[(edge[1], edge[0])] = weight

    nodes = set([node for edge in graph.keys() for node in edge])
    permutations = list(itertools.permutations(nodes))
    results = []
    for order in permutations:
        feedback_edges = calculate_feedback_arcset(graph, order)
        total_weight = sum(weight for edge, weight in feedback_edges)
        results.append((order, feedback_edges, total_weight))
    results.sort(key=lambda x: x[2])
    return results


def calculate_feedback_arcset_gurobi(preferences):
    new_preferences_dict3 = preferences.copy()
    for key, value in list(new_preferences_dict3.items()):
        if value == 1:
            new_preferences_dict3[(key[1], key[0])] = 1

    p = max(max(i, j) for i, j in new_preferences_dict3.keys()) + 1

    # Criar um novo modelo
    m = Model("feedback_arcset")

    # Criar variáveis
    x = {}
    for i in range(p):
        for j in range(i + 1, p):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Definir objetivo
    m.setObjective(sum(sum(new_preferences_dict3.get((k, j), 0) * x[k, j] for k in range(j)) +
                       sum(new_preferences_dict3.get((l, j), 0) * (1 - x[j, l]) for l in range(j + 1, p)) for j in range(p)), GRB.MINIMIZE)

    # Adicionar restrições
    for i in range(p):
        for j in range(i + 1, p):
            if i != j:
                for k in range(j + 1, p):
                    if k != i and k != j:
                        m.addConstr(x[i, j] + x[j, k] - x[i, k] <= 1)
                        m.addConstr(-x[i, j] - x[j, k] + x[i, k] <= 0)

    # Configurar o pool de soluções
    #m.setParam(GRB.Param.PoolSolutions, 1)  # Manter 5 soluções
    #m.setParam(GRB.Param.PoolSearchMode, 2)  # Encontrar as n melhores soluções

    # Otimizar
    m.optimize()

    # Criar um dicionário para armazenar as variáveis inversas e seus valores
    inverse_vars = {}

    results = []

    for i in range(m.SolCount):
        m.setParam(GRB.Param.SolutionNumber, i)
        feedback_edges = []

        for v in m.getVars():
            inverse_vars[(int(v.varName.split('_')[2]), int(v.varName.split('_')[1]))] = 1 - v.Xn

        for v in m.getVars():
            if (v.Xn == 1 and new_preferences_dict3.get((int(v.varName.split('_')[1]), int(v.varName.split('_')[2])), 0) != 0): 
                feedback_edges.append((int(v.varName.split('_')[1]), int(v.varName.split('_')[2])))
            elif (inverse_vars.get((int(v.varName.split('_')[2]), int(v.varName.split('_')[1])), 0) == 1 and new_preferences_dict3.get((int(v.varName.split('_')[2]), int(v.varName.split('_')[1])), 0) != 0):
                feedback_edges.append((int(v.varName.split('_')[2]), int(v.varName.split('_')[1])))

        total_weight = m.PoolObjVal
        results.append((feedback_edges, total_weight))

    results.sort(key=lambda x: x[1])
    return results


def add_edge(graph, u, v):
    graph[u].append(v)

def is_cyclic_util(graph, v, visited, parent):
    visited[v] = True
    for i in graph[v]:
        if visited[i] == False:
            if is_cyclic_util(graph, i, visited, v):
                return True
        elif parent != i and parent != -1 and i not in graph[parent]:  # Avoid cycles of length 2
            return True
    return False


def is_cyclic(graph, V):
    visited = [False] * (V)
    for i in range(V):
        if visited[i] == False:
            if is_cyclic_util(graph, i, visited, -1) == True:
                return True
    return False

def create_graph(edges):
    graph = defaultdict(list)
    for edge, weight in edges.items():
        u, v = edge
        add_edge(graph, u, v)
        if weight == 1:
            add_edge(graph, v, u)
    return graph
# Add a callback to update the hidden div whenever a radio button in the first column is clicked
def check_intransitivity(preferences):
    graph = create_graph(preferences)
    print (graph)
    # Create two graphs from the preferences
    if is_cyclic(graph, len(preferences)):

    
        # Convert each cycle back to a list and return the set of unique cycles
        return True
    else: 
        return False  # No intransitivity
    

def validate_preferences(preferences):
    # Verificar intransitividades
    has_intransitivity = check_intransitivity(preferences)  # Função check_intransitivity deve ser definida
    if has_intransitivity:
        results = calculate_feedback_arcset_gurobi(preferences)
        top_results = results[:1]
        print("TOPPPP", top_results)
        
        # Iterar sobre todos os resultados em top_results
        for result in top_results:
            # O primeiro elemento é a lista de pares de preferências
            first_result = result[0]
            # O segundo elemento é o valor da preferência
            #preference_value = result[1]
            
            # Iterar sobre cada par de preferências no primeiro elemento
            for old_key in first_result:
                # Determinar a nova chave
                new_key = (old_key[1], old_key[0])
                
                # Remover a entrada antiga, se existir
                if old_key in preferences:
                    del preferences[old_key]
                
                # Adicionar a nova entrada com o valor desejado
                preferences[new_key] = 2

        return preferences

    # Retornar "Perfect" se não houver erros encontrados
    return preferences

def matrix_to_preferences(matrix):
    num_criteria = matrix.shape[0]
    preferences_dict = {}
    
    for i in range(num_criteria):
        for j in range(i + 1, num_criteria):
            value = matrix[i, j]
            if value.is_integer():
                preferences_dict[(i, j)] = int(value)
            else:
                # Usar o valor da parte triangular inferior
                preferences_dict[(j, i)] = int(matrix[j, i])
    print(preferences_dict)
    return preferences_dict
def calculate_metrics(initial_matrix, final_matrix):
    delta = np.max(np.abs(final_matrix - initial_matrix))
    sigma = np.sqrt(np.sum((final_matrix - initial_matrix)**2) / initial_matrix.size)
    return delta, sigma

def our_methodology(matrix):
    num_criteria = matrix.shape[0]
    init_matrix = matrix.copy()
    preferences_dict = matrix_to_preferences(matrix)
    initial_CR = calculate_consistency_ratio(matrix, num_criteria)
    preferences_dict = validate_preferences(preferences_dict)
    weights, v, matrix1 = calculate_weights_and_v(num_criteria, preferences_dict)
    weights, matrices2, consistency_ratios = adjust_preferences(num_criteria, preferences_dict, weights, matrix1)

    #initial_matrix = np.array(matrices[0])
    final_matrix = np.array(matrices2[-1])
    
    
    delta, sigma = calculate_metrics(init_matrix, final_matrix)
    return initial_CR, final_matrix, delta, sigma, consistency_ratios[-1]
    
'''
matrices = {
   
    "Matrix 1 (ahpreductnovo)": np.array([
        [1, 5, 6, 7],
        [1/5, 1, 4, 6],
        [1/6, 1/4, 1, 4],
        [1/7, 1/6, 1/4, 1]
    ]),
    "Matrix 2 (redler4)": np.array([
        [1, 7, 1/5],
        [1/7, 1, 1/8],
        [5, 8, 1],
    ]),
    "Matrix 3 (redler 3)": np.array([
        [1, 2, 1/2, 2, 1/2, 2, 1/2, 2],
        [1/2, 1, 4, 1, 1/4, 1, 1/4,1 ],
        [2, 1/4, 1, 4, 1, 4, 1, 4],
        [1/2, 1, 1/4, 1, 1/4, 1, 1/4,1],
        [2, 4, 1, 4, 1, 4, 1, 4 ],
        [1/2, 1, 1/4, 1, 1/4, 1, 1/4, 1],
        [2, 4, 1, 4, 1, 4, 1, 4],
        [1/2,1,1/4,1,1/4,1,1/4,1]
    ]),
    "Matrix 4": np.array([
    [1, 2, 1/2, 2, 1/2, 2, 1/2, 2, 1/3],
    [1/2, 1, 4, 1, 1/4, 1, 1/4,1, 1/4 ],
    [2, 1/4, 1, 4, 1, 4, 1, 4,1/7],
    [1/2, 1, 1/4, 1, 1/4, 1, 1/4,1, 1/6],
    [2, 4, 1, 4, 1, 4, 1, 4, 6 ],
    [1/2, 1, 1/4, 1, 1/4, 1, 1/4, 1, 1/3],
    [2, 4, 1, 4, 1, 4, 1, 4, 7],
    [1/2,1,1/4,1,1/4,1,1/4,1,1/2],
    [3,4,7,6,1/6,3,1/7,2,1]
    
]),

    "Matrix 5 (redler4)": np.array([
    [1, 5, 3, 7, 6, 6, 1/3, 1/4],
    [1/5, 1, 1/3, 5, 3, 3, 1/5, 1/7],
    [1/3, 3, 1, 6, 3, 4, 6, 1/5],
    [1/7, 1/5, 1/6, 1, 1/3, 1/4, 1/7, 1/8],
    [1/6, 1/3, 1/3, 3, 1, 1/2, 1/5, 1/6],
    [1/6, 1/3, 1/4, 4, 2, 1, 1/5, 1/6],
    [3, 5, 1/6, 7, 5, 5, 1, 1/2],
    [4, 7, 5, 8, 6, 6, 2, 1]
]),
    "Matrix 6":np.array([
    [1, 1/9, 3, 1/5],
    [9, 1, 5, 2],
    [1/3, 1/5, 1, 1/2],
    [5, 1/2, 2, 1]
]),
    "Matrix 7":np.array([[1.,    2,   4.,    1/8],
 [1/2,    1.,    2.,    4.   ],
 [1/4,  1/2,   1.,    2  ],
 [8.,    1/4,  1/2,    1.,   ]]),

 "Matrix 8(inconred6)":np.array([[1.,    4,   3,    1, 3, 4],
 [1/4,    1,    7, 3, 1/5, 1 ],
 [1/3,1/7 , 1, 1/5, 1/5, 1/6  ],
 [1,1/3 , 5, 1, 1, 1/3],
 [1/3,5 , 5,1 , 1, 3],
 [1/4,1 , 6, 3, 1/3, 1]]),

 "Matrix 9(inconred6)": np.array([
    [1, 1/8, 1/3, 1/7, 1/3, 1, 8, 1/9, 1/4],
    [8, 1, 5, 1/2, 1/3, 4, 3,7, 5 ],
    [3, 1/5, 1, 2, 1/2, 1/6, 7, 7,1/9],
    [7, 2, 1/2, 1, 1, 5, 2,2, 1/9],
    [3, 3, 2, 1, 1, 7, 6, 5, 6 ],
    [1, 1/4, 6, 1/5, 1/7, 1, 2, 1/6, 1],
    [1/8, 1/3, 1/7, 1/2, 1/6, 1/2, 1, 1, 8],
    [9,1/7,1/7,1/2,1/5,6,1,1,1/8],
    [4,1/5,9,9,1/6,1,1/8,8,1]
    
]),
 "Matrix 10(inconred6)":np.array([
    [1, 8, 6, 1/2, 1/5],
    [1/8, 1, 7, 2, 7],
    [1/6, 1/7, 1, 5, 1/2],
    [2, 1/2, 1/5, 1, 1/2],
    [5, 1/7, 2, 2, 1]
]),

"Matrix 11(Balancing)":np.array([[1, 7, 9,5, 7, 5, 3],
 [1/7, 1, 5, 9, 5, 7, 5 ],
 [1/9,1/5 , 1, 7, 3, 7,3  ],
 [1/5,1/9 ,1/7, 1, 7, 5,5],
 [1/7,1/5 , 1/3,1/7 , 1, 9, 7],
 [1/5,1/7 , 1/7, 1/5, 1/9, 1, 5],
 [1/3,1/5 , 1/3, 1/5, 1/7, 1/5, 1]]),

 "Matrix 12(Consistent)":np.array([[1, 1/3, 1/5, 1, 1/4, 2, 3],
 [3, 1, 1/2, 2, 1/3, 3, 3 ],
 [5, 2 , 1, 4, 5, 6,5  ],
 [1,1/2 ,1/4, 1, 1/4, 1,2],
 [4,3 , 1/5, 4 , 1, 3, 1],
 [1/2,1/3 , 1/6, 1, 1/3, 1, 1/3],
 [1/3,1/3 , 1/5, 1/2, 1, 3, 1]]),

    "Matrix 13(detecting)":np.array([[1, 4,3,1,3,4],
 [1/4, 1, 7,3,1/5,1],
 [1/3,1/7,1,1/5,1/5,1/6],
 [1, 1/3, 5, 1, 1, 1/3],
 [1/3, 5, 5, 1, 1, 3],
 [1/4, 1, 6, 3, 1/3, 1]]),

 "Matrix 14(improving)":np.array([[1, 7,3,5,9,3,5],
 [1/7, 1, 3,3,5,3,3 ],
 [1/3,1/3 , 1,1/5,1/3,3,3  ],
 [1/5,1/3 ,5, 1, 9, 3,1/3],
 [1/9,1/5 , 3,1/9 , 1, 3, 1/5],
    [1/3,1/3 , 1/3, 1/3, 1/3, 1, 1/3],
    [1/5,1/3 , 1/3, 3, 5, 3, 1]]),
 
  
}


# Apply each technique to each matrix
results = {}

for matrix_name, matrix in matrices.items():
    results[matrix_name] = {}
    
    initial_CR, final_matrix, delta, sigma, final_CR = our_methodology(matrix)
    results[matrix_name]['Technique 1'] = {'Initial_CR': initial_CR,  'Final Matrix': final_matrix, 'Delta': delta, 'Sigma': sigma, 'CR': final_CR}
    
    
    

# Print the results
for matrix_name, techniques in results.items():
    print(f"\n{matrix_name}:")
    for technique, result in techniques.items():
        print(f"{technique} -> Initial_CR: {result['Initial_CR']:.4f}, Delta: {result['Delta']:.4f}, Sigma: {result['Sigma']:.4f}, CR: {result['CR']:.4f}")
        #print("Final Matrix:")
        #print(result['Final Matrix'])

'''

def generate_reciprocal_matrix(matrix):
    """Generate a reciprocal matrix based on the initial non-reciprocal upper triangle matrix."""
    n = matrix.shape[0]
    reciprocal_matrix = np.ones((n, n))  # Start with an identity matrix

    for i in range(n):
        for j in range(i + 1, n):
            reciprocal_matrix[i, j] = matrix[i, j]
            reciprocal_matrix[j, i] = 1 / matrix[i, j]

    return reciprocal_matrix

def generate_matrices(num_matrices, size_range=(3, 9)):
    """Generate a specified number of reciprocal matrices of random sizes."""
    matrices = {}

    for i in range(1, num_matrices + 1):
        # Randomly choose the size of the matrix
        size = np.random.randint(size_range[0], size_range[1] + 1)
        
        # Generate the initial random upper triangular matrix (excluding the diagonal)
        initial_matrix = np.ones((size, size))
        for row in range(size):
            for col in range(row + 1, size):
                initial_matrix[row, col] = np.random.randint(1, 10)
        
        reciprocal_matrix = generate_reciprocal_matrix(initial_matrix)
        matrices[f"Matrix {i}"] = reciprocal_matrix
    
    return matrices
# Parameters

# Generate 10,000 matrices
num_matrices = 10000
matrices = generate_matrices(num_matrices)

# Collect results
initial_CRs = []
final_CRs = []
deltas = []
sigmas = []
iter = 1
for matrix in matrices.values():
    
    
    initial_CR, final_matrix, delta, sigma, final_CR = our_methodology(matrix)
    initial_CRs.append(initial_CR)
    final_CRs.append(final_CR)
    deltas.append(delta)
    sigmas.append(sigma)
    iter += 1
    print(iter)

# Convert lists to numpy arrays for easier statistical analysis
initial_CRs = np.array(initial_CRs)
final_CRs = np.array(final_CRs)
deltas = np.array(deltas)
sigmas = np.array(sigmas)

# 1. Statistical Summary
print("Statistical Summary:")
print(f"Initial CR - Mean: {initial_CRs.mean():.4f}, Median: {np.median(initial_CRs):.4f}, Std: {initial_CRs.std():.4f}")
print(f"Final CR - Mean: {final_CRs.mean():.4f}, Median: {np.median(final_CRs):.4f}, Std: {final_CRs.std():.4f}")
print(f"Delta - Mean: {deltas.mean():.4f}, Median: {np.median(deltas):.4f}, Std: {deltas.std():.4f}")
print(f"Sigma - Mean: {sigmas.mean():.4f}, Median: {np.median(sigmas):.4f}, Std: {sigmas.std():.4f}")

# 2. Histogram of CR Values
plt.figure(figsize=(10, 5))
plt.hist(initial_CRs, bins=50, alpha=0.5, label='Initial CR')
plt.hist(final_CRs, bins=50, alpha=0.5, label='Final CR')
plt.xlabel('CR Value')
plt.ylabel('Frequency')
plt.title('Distribution of CR Values')
plt.legend()
plt.show()

# 3. Boxplots
plt.figure(figsize=(10, 5))
plt.boxplot([initial_CRs, final_CRs], labels=['Initial CR', 'Final CR'])
plt.ylabel('CR Value')
plt.title('Boxplot of Initial and Final CR')
plt.show()

# 4. Scatter Plot of Initial CR vs Final CR
plt.figure(figsize=(10, 5))
plt.scatter(initial_CRs, final_CRs, alpha=0.5)
plt.xlabel('Initial CR')
plt.ylabel('Final CR')
plt.title('Scatter Plot of Initial CR vs Final CR')
plt.show()

# 5. Threshold Analysis
threshold = 0.1
successful_matrices = np.sum(final_CRs < threshold)
print(f"Number of matrices with Final CR < {threshold}: {successful_matrices} / {num_matrices}")