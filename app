from gurobipy import Model, GRB
from math import factorial
import html
import dash
import sys
import dash_cytoscape as cyto
from sklearn.preprocessing import MinMaxScaler
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import numpy as np
import random
import signal
import math
import dash_bootstrap_components as dbc
import pandas as pd
import json
from plotly.graph_objects import Pie
import itertools
from math import factorial

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

value_changes = {
    1: [1],
    2: [3, 4, 5, 6],
    3: [2, 4, 5, 6, 7],
    4: [2, 3, 5, 6, 7],
    5: [ 2, 3, 4, 6, 7, 8],
    6: [ 3, 4, 5, 7, 8, 9],
    7: [ 3, 4, 5, 6, 8, 9],
    8: [ 4, 5, 6, 7, 9],
    9: [ 4, 5, 6,7, 8]
}


def calculate_weights_and_v(num_criteria, preferences):
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
    m.setParam(GRB.Param.PoolSolutions, 5)  # Manter 5 soluções
    m.setParam(GRB.Param.PoolSearchMode, 2)  # Encontrar as n melhores soluções

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





image1_path = 'assets/Captura de ecrã 2024-02-21, às 10.17.36.png'
image2_path ='assets/importance.png'
image3_path = 'assets/Captura de ecrã 2024-03-05, às 14.51.48.png'
image4_path = 'assets/2nd.png'
image5_path = 'assets/1st.png'
image6_path = 'assets/3.png'
image7_path = 'assets/pref.png'
image8_path = 'assets/4.png'
image9_path = 'assets/5.png'
image10_path = 'assets/6.png'
image11_path = 'assets/7.png'
image13_path ='assets/pair.png'
image12_path = 'assets/Imagem1.jpg'
image14_path = 'assets/once.png'

style_with_bg = {'width': '900px','height':'500px' ,'display': 'flex', 'justifyContent': 'space-around', 'backgroundImage': f'url({image12_path})', 'backgroundSize': '900px 500px', 'backgroundRepeat': 'no-repeat',  'margin': 'auto'}

# Function to generate a color scale
def generate_color_scale(base_color, n):
    blue_colors = [
        '#c8c4c4',  '#518585','#0f477f', 
        '#453e3e', '#0775ad', '#709fcd'
    ]
    
    orange_colors = [
        '#FFA500', '#993300', '#cc6633', '#cc9933', '#666633',
        '#FF6347', '#FFA07A', '#FA8072', '#E9967A', '#cc3300'
    ]
    
    if base_color == 'blue':
        return blue_colors[:n]
    elif base_color == 'red':
        return orange_colors[:n]
    else:
        raise ValueError("Base color must be 'blue' or 'red'")
selected_weights_stores = [dcc.Store(id={'type': 'selected-weights-store', 'index': i}, data=[]) for i in range(30)]


#### TOPSIS

def perform_topsis(values_matrix, criteria_types, weights):
    
    # Flatten the list of weights
    weights = [item for sublist in weights for item in sublist]
    weights = weights[0]
    print(weights)
    # Convert criteria types to weights for TOPSIS (benefit: 1, cost: -1)
    weights = [w if ct == 'benefit' else -w for w, ct in zip(weights, criteria_types)]
    
    # Perform TOPSIS step by step
    decision_matrix = np.array(values_matrix, dtype=float)
    # Step 1: Normalize the decision matrix
    norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    
    # Step 2: Calculate the weighted normalized decision matrix
    weighted_matrix = norm_matrix * weights
    print("weighted: ", weighted_matrix)
    # Step 3: Determine the ideal and negative-ideal solutions
    ideal_solution = np.max(weighted_matrix, axis=0)
    neg_ideal_solution = np.min(weighted_matrix, axis=0)
    # Step 4: Calculate the separation measures
    sep_measures = np.sqrt(((weighted_matrix - ideal_solution)**2).sum(axis=1))
    neg_sep_measures = np.sqrt(((weighted_matrix - neg_ideal_solution)**2).sum(axis=1))
    # Step 5: Calculate the relative closeness to the ideal solution
    closeness = neg_sep_measures / (sep_measures + neg_sep_measures)
    # Create ranking
    ranking = np.argsort(closeness)[::-1] + 1
    print(norm_matrix)
    print(weighted_matrix)
    print("closeness", closeness)
    print("ranking", ranking)

    return weighted_matrix, ideal_solution, neg_ideal_solution, sep_measures, neg_sep_measures, ranking



### divs






## Layout
app.layout = html.Div([
    #html.Div(id='title', className="info-box", children=[
     #   html.H2(" A gentle approach to multicriteria decision-making", style={"margin-top": "10px"}),
    #]),

    html.Div(id='content3', style={'display': 'none'}, children=[
    html.Div(id='title', className="info-box", children=[
        html.H2(" A gentle approach to multicriteria decision-making", style={"margin-top": "10px"}),
    ]),    

    html.Hr(),

    html.Div([
        html.Div([
            
            html.Button('Criteria and alternatives', id='insertinfo-button', n_clicks=0, className='btn btn-outline-primary custom-button3'), html.Div(id='boxes', style={'display': 'none'}, className='boxtot')
        ], className='three columns'),
        


        html.Div([
            
            html.Button('Preferences between criteria', id='see-preferences-button', className='btn btn-outline-primary custom-button3'),
            dcc.Graph(id='plot', style={'display': 'none'}, className='barplot'),], className='three columns'),
        
        
        #html.Div([ html.Button('Consistency ratio', id='perform-button', n_clicks=0, className='btn btn-outline-primary custom-button3'), dcc.Graph(id='line-chart', figure=go.Figure(), style={'display': 'none'}),], className='four columns'),
        
        
        html.Div([html.Button('Weights associated', id='see-weights-button', n_clicks=0, className='btn btn-outline-primary custom-button3'), html.Div(id='weights-graph-container')], className='three columns'),
           
    ], className='row'),

    #html.Hr(),
 

    html.Div([
        html.Div([
            
            html.Button('Distance from the ideal solution', id='distance-button', n_clicks=0, className='btn btn-outline-primary custom-button3'), html.Div(id='distance-result',style={'padding-left': '50px'})
        ], className='three columns'),
        


        html.Div([
             
            html.Button('Alternatives performance on each criteria', id='see-performance-button', className='btn btn-outline-primary custom-button3'), html.Div(id= 'performance-result')
            ], className='three columns'),
        
        
        html.Div([ html.Button('Alternatives ranking', id='ranking-button', n_clicks=0, className='btn btn-outline-primary custom-button3'), html.Div(id='ranking-result')], className='three columns'),
        
        
        
    ], className='row1'),

    html.Hr(),

    ]),
    
    html.Div(id='content', style={'display': 'none'}, children=[

    

    
    html.Div(id='intro', className="info-box", children=[    
        html.H2("A gentle approach to multicriteria decision-making"),

        html.Br(),
        html.Br(),

        dbc.Row([
        dbc.Col([  
            

            html.H4("Once upon a time"),
            
            html.Br(),
            
            html.Img(src=image14_path , style={"width": "350px"}),
            
        ], width=3, className='my-col-4'),
        dbc.Col([       
        html.Br(),
        html.Br(),                                               
        html.P(" In the busy world of our company, important decisions pop up all the time, and we have just the thing to help us make the right choices. It's a special tool that's like a guide through the labyrinth of decision-making. We call it Multi-Criteria Decision Analysis (MCDA). It's not as complicated as it sounds! This tool uses smart methods, such as the Analytic Hierarchy Process (AHP) and the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS). These methods are like magic because they let us look at lots of important stuff all at once when making decisions. It's kind of like how our brains juggle lots of thoughts before making up our minds. With this tool, we can tackle tough decisions with confidence and make sure we're heading in the right direction. ",
    style={'textAlign': 'justify-left'}),
        html.P("AHP breaks down the decision into smaller parts and helps us figure out what's most important. Then TOPSIS ranks our options, so we know which one is closest to being perfect. Together, they give us a clear and logical way to make the best decision possible. Are you ready to start your decision-making journey? Let's go!",
    style={'textAlign': 'justify-left'})
    ], width=9, className='my-col-8')]),
    ], style={'display': 'block'}),  

    
    #criteria
    html.Div(id='problem-structuring', className="info-box", children=[
    html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
    html.H6("Problem Structuring"),
    html.H4("Tell me about the situation you are facing"),
    dbc.Row([
        dbc.Col([  
            
            html.Br(),
            
            
            
            
            html.Img(src=image5_path , style={"width": "350px"}),
            
        ], width=3, className='my-col-4'),
        dbc.Col([ 
            html.Br(),
             
            html.P("Embarking on this decision-making journey is like preparing for a mountain climb. The first and most crucial step is understanding the situation you’re facing. Imagine standing at the foot of a mountain, looking up at the peak. The peak represents the best possible outcome - the solution that serves your specific target. But to reach the peak, you first need to understand the terrain, the possible paths, and the challenges you might face. That’s exactly what we’re doing here. So, let’s get started. Fill in the details below and embark on the first step of your decision-making journey. Remember, every great journey begins with a single step. ",
    style={'textAlign': 'justify-left'}), 
            dbc.InputGroup([
                dcc.Input(id='decision-makers', type='text', placeholder='The decision makers', style={'width': '200px'}),

                html.P(" need to make a decision about ", className='custom-p'),
                dcc.Input(id='situation-to-solve', type='text', placeholder='situation to solve' , style={'width': '25px !important'}),
                #html.P(".", className='custom-p'),
            ], className="mb-3"),
            dbc.InputGroup([
                
                dcc.Input(id='decision-makers-2', type='text', placeholder='The decision makers',   style={'width': '200px'}),
                html.P(" are concerned with ", className='custom-p'),
                dcc.Input(id='how-the-situation', type='text', placeholder='how it will impact on the several areas',   style={'width': '350px'}),
                #html.P(".", className='custom-p'),
            ], className="mb-3"),
            
            dbc.InputGroup([
                html.P("The goal is: ", className='custom-p2'),
                dcc.Input(id='goal', type='text', placeholder='situation to solve',   style={'width': '200px'}),
                html.P(" that would best serve ", className='custom-p2'),
                dcc.Input(id='specific-target', type='text', placeholder='a specific target',   style={'width': '45px !important'}),
                #html.P(".", className='custom-p'),
    ], className="mb-3"),
], width=9, className='my-col-8')]),
    ], style={'display': 'none'}),


    

    html.Div(id='criteriabox', className="info-box", children=[
    html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
    html.H4("What criteria are important to you in choosing the right path?"),     
    #html.H4("What criteria or considerations are important to you in choosing the right path?"),    
    dbc.Row([
        dbc.Col([
            
             
            html.Br(), 
            html.Img(src=image4_path , style={"width": "350px"}),
            
        ], width=3, className='my-col-4'),
        dbc.Col([
                
            html.Br(),
            html.P("Every decision needs some criteria to evaluate the alternatives you have. The criteria can be of two different types, Benefit or Cost. 'Benefit' is when we want the criterion to have a positive impact on the evaluation of alternatives, and 'Cost' is when we want the impact to be negative. For example, in a decision-making problem, a criterion like 'Profit' could be classified as a 'Benefit' because a higher profit is desirable. On the other hand, a criterion like 'Expenses' could be classified as a 'Cost' because a higher expense is undesirable.",
    style={'textAlign': 'justify-left'}),
            dbc.InputGroup([
            html.P("How many criteria are you considering?", className='custom-p'), 
            dcc.Input(id='num-criteria', type='number', placeholder='Insert a number (3-10)', style={"width": "200px", "height":"25px", "margin-top":"20px", "margin-left":"5px"}),
            html.Button('Submit', id='submit-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
            ]),
               
                html.Div(id='criteria-table-container', style={'display': 'none'}, children=[
                    html.P('Enter names for each criterion:'),
                    html.Table(id='criteria-table'),
                    html.Button('Save', id='save-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
                 
            ]),
        ], width=9, className='my-col-8'),
    ]),
], style={'display': 'none'}),
    
        
    dcc.Store(id='criteria-names'),
    dcc.Store(id='criteria-types'),

    
    html.Div(id='criteriabox2',className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("How would you rank these criteria in order of importance?"),
        dbc.Row([
        dbc.Col([
        
        html.Br(),    
        html.Img(src=image6_path , style={"width": "350px"}),
        ], width=3, className='my-col-4'),
        dbc.Col([ 
            html.Br(),
        html.P("As we continue our journey up the mountain, we come to a crucial juncture - selecting preferences. This is where we decide which criteria are most important to us. Imagine you’re at a fork in the trail on your mountain climb. Both paths lead to the top, but each one offers a different experience. One path might be steeper but offers a quicker ascent, while the other is longer but has a gentler slope. Which path you choose depends on what’s important to you - speed or ease of climb.",
    style={'textAlign': 'justify-left'}),
        html.P([
    "None: Criteria have equal importance", html.Br(),
    "2: Equal to moderately more importance", html.Br(),
    "3: Moderately more importance", html.Br(),
    "4: Moderately to strongly more importance",html.Br(),
    "5: Strongly different more importance",html.Br(),
    "6: Strongly to very strongly more importance" ,html.Br(),
    "7: Very  more importance" ,html.Br(),
    "8: Very strongly to extremely more importance" ,html.Br(),
    "9: Extremely more importance",html.Br(),
]),

       
        
    #html.Button('Select', id='select-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
    html.Div(id='hidden-div-2', style={'display': 'none'}),

    
    ], width=9, className='my-col-8'),
    ]),
    html.Div(id='preferences-dropdown',style={'display': 'flex', 'justifyContent': 'center'}),
    

    dcc.Store(id='hidden-div', data=[]),

    html.Div(
        html.Button('Validate', id='validate-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
        style={'display': 'flex', 'justifyContent': 'center'}
    ),

    html.Div(id="error-message", style={"color": "black", "font-weight": "bold", "text-align":"center"}),
    ], style={'display': 'none'}),

   
    html.Div(id='criteriafinal',className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("If you’re unsure about the criteria, could you discard any of them?"),
        dbc.Row([
        dbc.Col([
            html.Br(),
            html.Img(src=image8_path , style={"width": "350px"}),
            ], width=3, className='my-col-4'),
        dbc.Col([ 
            html.Br(),
            
    html.P("If you’re unsure about any of the criteria, it’s worth asking: Could you discard any of them? This is like looking at a signpost and deciding whether it’s helpful for your journey. Remember, the criteria you choose should accurately represent your problem and what’s important to you. They should be distinct and not too similar to each other. Choosing criteria that are too similar can make the process more complicated and redundant, like having two signposts pointing in the same direction. So, take a moment to reflect on your criteria. Are they all necessary? Do they accurately represent your problem? Are any of them too similar?", style={'textAlign': 'justify-left'}),        
    html.Button('Current problem', id='confirm-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
    html.Div(
    children=[
        cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'breadthfirst', 'autoungrabify': True, 'zoomingEnabled': False},

        style={'width': '100%', 'height': '200px', 'margin':'0, auto'},
        elements=[],
        panningEnabled=True,  # Disables user panning
        autolock=False,  
    ),
    ],
    style={'width': '80%','margin':'0 auto' }
),
    ], width=9, className='my-col-8'),
    ]),
    ], style={'display': 'none'}),
    #alternatives
    
    html.Div(id='alternativebox', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("What are the different paths you could take to overcome this challenge?"),
        dbc.Row([
        dbc.Col([
          
        html.Br(),  
        html.Img(src=image9_path , style={"width": "350px"}),
        ], width=3, className='my-col-4'),
        dbc.Col([ 
            html.Br(),
        html.P("Criteria serve as the backbone of our decision-making process. They are employed to assess the diverse alternatives that may exist to solve a problem. Each alternative is evaluated based on these criteria, allowing us to compare and contrast the different options.",
    style={'textAlign': 'justify-left'}),
        dbc.InputGroup([
        html.P("How many alternatives are you considering?", className='custom-p'),
        dcc.Input(id='num-alternatives', type='number', placeholder='Insert a number (2-10)', style={"width": "200px", "height":"25px", "margin-top":"20px",  "margin-left":"5px"}),
        html.Button('Submit', id='submit-button2', n_clicks=0, className='btn btn-outline-primary custom-button'),
        ]),
        html.Div(id='alternatives-table-container', style={'display': 'none'}, children=[
        html.P('Enter names for each alternative:'),
        html.Table(id='alternatives-table', className='center-table2'),
        html.Button('Save', id='save-button2', n_clicks=0, className='btn btn-outline-primary custom-button'),
    ]),
    dcc.Store(id='alternatives-names'),
        ], width=9, className='my-col-8'),
    ]),  
    ], style={'display': 'none'}),
    

   
    html.Div(id='insertvalues', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        
        html.H4("What kind of data or information do you have available for each alternative with respect to each criterion?"),
        dbc.Row([
        dbc.Col([
        html.Br(),    
        html.Img(src=image10_path , style={"width": "350px"}),
        ], width=3, className='my-col-4'),
        dbc.Col([ 
        html.Br(),
        html.P("Now that you have your criteria and identified your alternatives, it is time to quantify the decision-making process. Here you will fill the table with the performance values of each alternative for each criterion. These values can be based on data, estimates, or even subjective judgements, depending on the nature of the decision problem. ",
    style={'textAlign': 'justify-left'}),
    html.Button('Insert', id='insert-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
        html.Div(id='values-table-container', style={'display': 'none'}, children=[
        html.P('Enter values for each alternative/criterion:'),
        html.Table(id='values-table', style={"width": "600px"})
    ]),
        html.Button('Save values', id='save-values-button', n_clicks=0, className='btn btn-outline-primary custom-button'),

        dcc.Store(id='values-matrix'),

    ], width=9, className='my-col-8'),
    ]),
         
    ], style={'display': 'none'}),

    
    html.Div(id='alternativeboxsure', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("If you’re unsure about the paths, is there any that you can eliminate right from the start?"),
        dbc.Row([
        dbc.Col([
        html.Br(),    
        html.Img(src=image11_path , style={"width": "350px"}),
        ], width=3, className='my-col-4'),
        dbc.Col([ 
            html.Br(),
        html.P("You see several paths leading up the mountain. But upon closer inspection, you realize that some paths are blocked by large boulders or lead to dead ends. These are paths you can eliminate right away. Similarly, in our decision-making process, if there are alternatives that you know won't work for your situation, it's okay to discard them from the start. This can simplify your decision-making process and make it easier to focus on the viable alternatives. The next graph will give you a summary of your decision problem so take a closer look to see if everything is as you want it.",
    style={'textAlign': 'justify-left'}),
        #html.Button('Show graph', id='hierarchy-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
    html.Div(id='graph-container'),
        html.Button('Proceed', id='saveinfo-button', n_clicks=0, className='btn btn-outline-primary custom-button' ), 
        ], width=9, className='my-col-8'),
    ]),
     
    ], style={'display': 'none'}),
    

    html.Button('Back', id='back-button', n_clicks=0, className='btn btn-dark custom-button2'),
    html.Button('Next', id='next-button', n_clicks=0, className='btn btn-dark custom-button2'),
    dcc.Store(id='current-div', data='intro'),

    
    
    
    
    ]),    
    
    html.Div(id='content-2', children=[
    html.Div(id='explanation-box', className="explanation-box", children=[
    html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
    html.H4("AHP"),
    html.P("The Analytic Hierarchy Process (AHP) is a method of multi-criteria decision-making that uses mathematics and psychology. It was developed by Thomas L. Saaty in the 1970s. The method is based on the establishment of hierarchies of criteria, which are then evaluated separately, and the results are combined into a single score. Let’s dive into each step.",
    style={'textAlign': 'justify-left'}),
    
], style={'display': 'block'}),
    
    
    html.Div(id='ahpbox', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("Generate Results"),
        html.P("In the AHP, the principal eigenvalue (the largest eigenvalue) of the pairwise comparison matrix is used to calculate the weights of the criteria. The corresponding eigenvector is normalized to sum to one and gives the weights of the criteria. The reason for using the principal eigenvalue and its corresponding eigenvector is that they provide the best approximation of the weights of the criteria, given the pairwise comparisons. This is based on the Perron-Frobenius theorem, which states that a positive square matrix has a unique largest real eigenvalue and that the corresponding eigenvector can be chosen to have strictly positive components. This makes it suitable for deriving weights in the AHP.",
    style={'textAlign': 'justify-left'}),
    html.Button('Generate', id='generate-button', n_clicks=0, className='btn btn-outline-primary custom-button'),
        
html.Div(id='tres',className="info-box", children=[
    
    html.Div([
        
        html.Br(),
        html.H6('Your Comparison Matrix', className='suggestion'),
        html.Div(
            dash_table.DataTable(
                id='comparison-matrix',
                columns=[],
                style_table={'width': '150px'},
                style_header={'display': 'none'}
            ), style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'flex-start'}
        )
    ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        
        html.Br(),
        html.H6('Largest Eigenvalue ', className='suggestion'),
        html.Br(), 
        html.Div(id='max-eigenvalue')
    ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        
        html.Br(),
        html.H6('Principal Eigenvector', className='suggestion'),
        html.Br(), 
        html.Pre(id='principal-eigenvector')
    ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
])

    ], style={'display': 'none'}),
    
    
    
    html.Div(id='consistency', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("Are your preferences consistent?"),
        html.P("The consistency ratio is used to check the consistency of the pairwise comparisons. If the pairwise comparison matrix was perfectly consistent, then it would be reciprocal (i.e., if criterion A is twice as important as criterion B, then criterion B is half as important as criterion A), and all its eigenvalues would be equal to the number of criteria.The consistency index (CI) measures the deviation from perfect consistency. It’s calculated as (λ_max - n) / (n - 1), where λ_max is the largest eigenvalue, and n is the number of criteria. The consistency ratio (CR) is then calculated as CI divided by the random index (RI), which is the average consistency index for a randomly generated reciprocal matrix. If the CR is less than 0.1, the judgments are considered to be acceptably consistent.",
    style={'textAlign': 'justify-left'}),
    html.Br(),
    html.Div([
        html.P('Your consistency ratio'),
        html.Div(id='consistency-ratio'),]),
          
    ], style={'display': 'none'}),
    
        
    html.Div(id='consistency-choice', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("Aren't your preferences consistency enough?"),
        html.P("If the consistency ratio of the preferences provided by you is bigger than 0,1 we fine-tune the decision matrix to make it more consistent. We make small adjustments to the preferences while preserving their logic, aiming to achieve a more consistent matrix. ",
    style={'textAlign': 'justify-left'}),
        html.P("On the dashboard, you’ll see a bar chart showing the consistency ratio for each iteration. You can click on two bars to compare the iterations and decide which one aligns best with your intuition. This way, you have control over the fine-tuning process and can choose the path that feels right to you.", style={'textAlign': 'justify-left'}),
        html.Br(),
        dcc.Graph(id='consistency-ratio-graph', figure=go.Figure()),  # Replace html.Div with dcc.Graph
    dcc.Store(id='comparison-matrices-store'), 
    dcc.Store(id='selected-bars-store', data=[]), 
    dcc.Store(id='clicked-heatmap-indices-store', data=[]), 
    dcc.Store(id='weights-store'),  
    dcc.Store(id='matrix-store'),  
    dcc.Store(id='consistency-ratios-store'),  
    
    html.Div(id='comparison-matrix-heatmaps', children=[]),  # Add an empty div to hold the button
    
    html.Button('Choose iteration', id='choose-iteration-button', n_clicks=0, className='btn btn-outline-primary custom-button', style={'display': 'none'}),  # Add the button outside of any callbacks with display: none

    dbc.Row(id='units-container', children=[], style={"display":"flex", "flex-wrap": "nowrap"}), 
    html.Div(
    html.Button('Proceed', id='hide-button', n_clicks=0, style={'display': 'none'} , className='btn btn-outline-primary custom-button' ),
    style={'display': 'flex', 'justify-content': 'center'}),

    ], style={'display': 'none'}),
    
    html.Button('Back', id='back-button2', n_clicks=0, className='btn btn-dark custom-button2'),
    html.Button('Next', id='next-button2', n_clicks=0, className='btn btn-dark custom-button2'),
    dcc.Store(id='current-div2', data='explanation-box'),

    
    ], style={'display': 'none'}),

   
    html.Div(id='content-3', children=[
    html.Div(id='topsis', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
    html.H4("Topsis"),
    html.P("The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) is a method of multi-criteria decision-making that uses mathematics and compensatory aggregation. It was originally developed by Ching-Lai Hwang and Yoon in 19811. The method is based on the concept that the chosen alternative should have the shortest euclidean distance from the positive ideal solution (PIS) and the longest euclidean distance from the negative ideal solution (NIS).",
    style={'textAlign': 'justify-left'}),], style={'display': 'block'}),
    
    
    
    html.Div(id='topsis2box', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("What happens here?"),
        html.P("So far, we have worked with the criteria and assigned a weight to each of them. Now, we will also work with the alternatives to establish a ranking for them. To do this, we will first normalize the matrix values and determine the ideal best and worst values. How? For each criterion, we will identify the best and worst existing alternatives. Please review your matrix",
    style={'textAlign': 'justify-left'}),
        

        
        
        
        html.Button('Your best and worst alternatives', id='see-values-matrix-button', className='btn btn-outline-primary custom-button'),
        html.Div(
    children=[
        html.Div(id='values-matrix-table')
    ],
    style={'width': '65%','margin':'0 auto' }
),
    



    ], style={'display': 'none'}),
    
    html.Div(id='topsisconclusion', className="info-box", children=[
        html.H2("A gentle approach to multicriteria decision-making"),
    html.Br(),
    html.Br(),
        html.H4("What now?"),
        html.P("We’ve reached the summit of our decision-making journey. The view from here is clear and we can see the landscape of our decision problem. The TOPSIS analysis has been performed and it’s time to interpret the results.", style={'textAlign': 'justify-left'}),

        html.P("First, you’ll see a Distance Graph. This graph shows the distance of each alternative to the ideal solution and to the worst solution. It’s like looking at how far each path has taken us from the start of our climb and how close we are to the peak.", style={'textAlign': 'justify-left'}),

        html.P("Next, there’s a Radar Chart. This chart allows you to see the performance of each alternative on each criterion. It’s like having a bird’s eye view of the different paths, showing us the twists and turns each one took.", style={'textAlign': 'justify-left'}),

        html.P("Finally, there’s a Pyramid Chart. This chart presents the final ranking of the alternatives. It’s like a beacon at the peak of the mountain, shining a light on the path that has led us to the best decision.", style={'textAlign': 'justify-left'}),

        html.P("Thank you for embarking on this journey with us. We hope that this process has provided you with valuable insights and guided you towards the best decision. ",
    style={'textAlign': 'justify-left'}),

        html.Div(
    html.Button('Proceed', id='hide2-button', n_clicks=0, style={'display': 'none'}, className='btn btn-outline-primary custom-button' ),
    style={'display': 'flex', 'justify-content': 'center'}),
        
    ], style={'display': 'none'}),


    
    
    html.Button('Back', id='back-button3', n_clicks=0, className='btn btn-dark custom-button2'),
    html.Button('Next', id='next-button3', n_clicks=0, className='btn btn-dark custom-button2'),
    dcc.Store(id='current-div3', data='topsis'),], style={'display': 'none'})
    

]+ selected_weights_stores)

# Callback to hide the content and show the boxes
@app.callback(
    [Output('content', 'style'), Output('content3', 'style'), Output('boxes', 'style'), Output('boxes', 'children'), Output('content-2', 'style'), Output('hide-button', 'style'), Output('content-3', 'style'), Output('hide2-button', 'style')],
    [Input('insertinfo-button', 'n_clicks'), Input('saveinfo-button', 'n_clicks'), Input('see-preferences-button', 'n_clicks'), Input('hide-button', 'n_clicks'), Input('distance-button', 'n_clicks'), Input('hide2-button', 'n_clicks')],
    [State('criteria-names', 'data'), State('alternatives-names', 'data')]
)
def toggle_content_boxes(n1, n2, n3, n4, n5, n6, criteria_names, alternatives_names):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none', 'width': '60%'}, None, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'insertinfo-button' and n1 > 0:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none', 'width': '60%'}, None, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        elif button_id == 'saveinfo-button' and n2 > 0:
            # Generate color scales
            criteria_colors = generate_color_scale('blue', len(criteria_names))
            alternatives_colors = generate_color_scale('red', len(alternatives_names))

            # Create a single Div for all criteria and a single Div for all alternatives
            criteria_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': criteria_colors[i], 'border-radius': '30%', 'display': 'inline-block','margin-right': '10px'}), f'C{i+1}: {name}'], className='box criteria') for i, name in enumerate(criteria_names)]
            alternatives_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': alternatives_colors[i], 'border-radius': '30%', 'display': 'inline-block', 'margin-right': '10px'}), f'A{i+1}: {name}'], className='box alternatives') for i, name in enumerate(alternatives_names)]

            # Add these Divs to your layout
            boxes = html.Div(children=[
                html.Div('Criteria:', className='header1'),
                html.Div(criteria_divs, className='box-container'),

                html.Div('Alternatives:', className='header'),
                html.Div(alternatives_divs, className='box-container'),
            ])

            return {'display': 'none'}, {'display': 'block'}, {'display': 'block', 'width': '80%'}, boxes, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        elif button_id == 'see-preferences-button' and n3 > 0:
            return dash.no_update, {'display': 'none'}, dash.no_update, None, {'display': 'block'}, {'display': 'block'}, dash.no_update, dash.no_update
        elif button_id == 'hide-button' and n4 > 0:
            criteria_colors = generate_color_scale('blue', len(criteria_names))
            alternatives_colors = generate_color_scale('red', len(alternatives_names))

            # Create a single Div for all criteria and a single Div for all alternatives
            criteria_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': criteria_colors[i], 'border-radius': '50%', 'display': 'inline-block','margin-right': '10px'}), f'C{i+1}: {name}'], className='box criteria') for i, name in enumerate(criteria_names)]
            alternatives_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': alternatives_colors[i], 'border-radius': '50%', 'display': 'inline-block', 'margin-right': '10px'}), f'A{i+1}: {name}'], className='box alternatives') for i, name in enumerate(alternatives_names)]

            # Add these Divs to your layout
            boxes = html.Div(children=[
                html.Div('Criteria:', className='header1'),
                html.Div(criteria_divs, className='box-container'),

                html.Div('Alternatives:', className='header'),
                html.Div(alternatives_divs, className='box-container'),
            ])
            return dash.no_update, {'display': 'block'}, {'display': 'block', 'width': '80%'}, boxes,{'display': 'none'}, {'display': 'none'}, dash.no_update, dash.no_update

        elif button_id == 'distance-button' and n5 > 0:
            return dash.no_update, {'display': 'none'}, dash.no_update, None, dash.no_update, dash.no_update, {'display': 'block'}, {'display': 'block'}
        elif button_id == 'hide2-button' and n6 > 0:
            criteria_colors = generate_color_scale('blue', len(criteria_names))
            alternatives_colors = generate_color_scale('red', len(alternatives_names))

            # Create a single Div for all criteria and a single Div for all alternatives
            criteria_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': criteria_colors[i], 'border-radius': '50%', 'display': 'inline-block','margin-right': '10px'}),  f'C{i+1}: {name}'], className='box criteria') for i, name in enumerate(criteria_names)]
            alternatives_divs = [html.Div([html.Span(style={'height': '8px', 'width': '8px', 'background-color': alternatives_colors[i], 'border-radius': '50%', 'display': 'inline-block', 'margin-right': '10px'}), f'A{i+1}: {name}'], className='box alternatives') for i, name in enumerate(alternatives_names)]

            # Add these Divs to your layout
            boxes = html.Div(children=[
                html.Div('Criteria:', className='header1'),
                html.Div(criteria_divs, className='box-container'),

                html.Div('Alternatives:', className='header'),
                html.Div(alternatives_divs, className='box-container'),])
            return dash.no_update, {'display': 'block'}, {'display': 'block', 'width': '80%'}, boxes, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    return dash.no_update, dash.no_update, dash.no_update, None, dash.no_update, dash.no_update, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('criteria-table-container', 'style'),
    Output('criteria-table', 'children'),
    Input('submit-button', 'n_clicks'),
    State('num-criteria', 'value')
)
def update_criteria_table_visibility(n_clicks, num_criteria):
    print("Entered update_criteria_table_visibility function")
    if n_clicks > 0:
        table_rows = []
        for i in range(num_criteria):
            row = [html.Td(f'Criterion {i+1}:', className='suggestion'), html.Td(dcc.Input(id={'type': 'criteria-name', 'index': i}, type='text')),
                   html.Td(dcc.Dropdown(id={'type': 'criteria-type', 'index': i}, options=[{'label': 'Benefit', 'value': 'benefit'}, {'label': 'Cost', 'value': 'cost'}], value='benefit', className='drop-criteria'))]
            table_rows.append(html.Tr(row))
        return {'display': 'block'}, table_rows
    return {'display': 'none'}, []

@app.callback(
    [Output('alternatives-table-container', 'style'),
     Output('alternatives-table', 'children')],
    Input('submit-button2', 'n_clicks'),
    State('num-alternatives', 'value')
)
def update_alternatives_table_visibility(n_clicks, num_alternatives):
    print("Entered update_alternatives_table_visibility function")
    if n_clicks > 0:
        table_rows = []
        for i in range(num_alternatives):
            row = [html.Td(f'Alternative {i+1}:', className='suggestion'), html.Td(dcc.Input(id={'type': 'alternative-name', 'index': i}, type='text'))]
            table_rows.append(html.Tr(row))
        return {'display': 'block'}, table_rows
    return {'display': 'none'}, []




@app.callback(
    Output('criteria-names', 'data'),
    Output('criteria-types', 'data'),
    Input('save-button', 'n_clicks'),
    [State({'type': 'criteria-name', 'index': ALL}, 'value'),
     State({'type': 'criteria-type', 'index': ALL}, 'value')]
)
def save_criteria_names(n_clicks, names, types):
    print("Entered save_criteria_names function")
    if n_clicks > 0:
        return names, types
    return None, None

@app.callback(
    Output('alternatives-names', 'data'),
    Input('save-button2', 'n_clicks'),
    [State({'type': 'alternative-name', 'index': ALL}, 'value')]
)
def save_alternatives_names(n_clicks, values):
    print("Entered save_alternatives_names function")
    if n_clicks > 0:
        return values
    return None





@app.callback(
    [Output('values-table-container', 'style'),
     Output('values-table', 'children')],
    Input('insert-button', 'n_clicks'),
    State('num-alternatives', 'value'),
    State('num-criteria', 'value'),
    State('criteria-names', 'data'),
    State('alternatives-names', 'data')
)
def update_values_table_visibility(n_clicks, num_alternatives, num_criteria, criteria_names, alternatives_names):
    print("Entered update_values_table_visibility function")
    if n_clicks > 0 and criteria_names and alternatives_names:
        table_rows = [
            html.Tr([html.Th('  ', className='suggestion')] + [html.Th(name, className='suggestion') for name in criteria_names])
        ]
        for i in range(num_alternatives):
            row = [html.Th(alternatives_names[i], className='suggestion alt-name-cell')]
            for j in range(num_criteria):
                row.append(html.Td(dcc.Input(id={'type': 'value', 'index': f'{i}-{j}'}, type='text', style={'width': '100px'})))
            table_rows.append(html.Tr(row))
        return {'display': 'block'}, table_rows
    return {'display': 'none'}, []



@app.callback(
    Output('hidden-div-2', 'children'),
    [Input({'type': 'dynamic-radio', 'index': ALL}, 'value')]
)
def update_hidden_div_2(values):
    # Store the values of the radio buttons in the first column
    first_column_values = [value for value in values if value is not None and value.endswith('-1')]
    return first_column_values

# Modify the update_dropdown function to access the data in the hidden div
@app.callback(
    Output('preferences-dropdown', 'children'),
    [Input('criteria-names', 'data'), Input('criteria-types', 'data')],
    [State('hidden-div-2', 'children')]
)
def update_dropdown(criteria_names, types, radio_values):
    if criteria_names and types:
        num_criteria = len(criteria_names)
        rows = []
        for i in range(num_criteria):
            for j in range(i + 1, num_criteria):
                row = html.Tr([
                    html.Td(dcc.Checklist(
                        options=[
                            {'label': criteria_names[i], 'value': f'{i}-{j}-i'},
                            {'label': criteria_names[j], 'value': f'{i}-{j}-j'}
                        ],
                        id={'type': 'dynamic-checklist', 'index': f'{i}-{j}-importance'},
                        inline=True
                    )),
                    html.Td(dcc.Checklist(
                        options=[
                            {'label': str(k), 'value': f'{i}-{j}-{k}'} for k in range(2, 10)
                        ],
                        id={'type': 'dynamic-checklist', 'index': f'{i}-{j}-k'},
                        inline=True
                    )),
                    html.Td(dcc.Checklist(
                        options=[{'label': '', 'value': f'{i}-{j}-equal'}],
                        id={'type': 'dynamic-checklist', 'index': f'{i}-{j}-equal'},
                        inline=True,
                        style={'padding-left': '20px'}
                    )),
                ])
                rows.append(row)

        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Which of these criteria is more important than the other?', className='suggestion', style={'padding-right': '20px'}),
                    html.Th('How much more?', className='suggestion', style={'padding-right': '40px'}),
                    html.Th('None is more important', className='suggestion', style={'padding-left': '20px'})
                ])
            ]),
            html.Tbody(rows, className='suggestion')
        ], className='suggestion')
        return table
    return []
from collections import defaultdict

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
    

def validate_preferences(criteria_names, preferences, num_criteria):
    expected_comparisons = factorial(num_criteria) // (2 * factorial(num_criteria - 2))

    # Verificar comparações faltantes
    selected_comparisons = len(preferences)
    if selected_comparisons != expected_comparisons:
        return "There are missing or duplicate comparisons for some criteria"

    # Verificar comparações duplicadas
    unique_pairs = set(preferences.keys())
    if selected_comparisons >= expected_comparisons and len(unique_pairs) < selected_comparisons:
        return "Duplicate comparisons for some criteria"

    # Verificar intransitividades
    has_intransitivity = check_intransitivity(preferences)  # Função check_intransitivity deve ser definida
    if has_intransitivity:
        results = calculate_feedback_arcset_gurobi(preferences)
        top_results = results[:5]

        all_suggestions = []
        for i, result in enumerate(top_results, 1):
            feedback_edges = result[0]
            suggested_changes = [
                html.Li(
                    f"Change preference {criteria_names[edge[0]]} - {criteria_names[edge[1]]} to {criteria_names[edge[1]]} > {criteria_names[edge[0]]}", 
                    className='suggestion'
                )
                for edge in feedback_edges
            ]
            all_suggestions.append(html.Div([
                html.H4(f"Suggestion {i}:", className='suggestion'),
                html.Ul(suggested_changes, className='no-bullets')
            ]))

        return html.Div(all_suggestions, className='suggestion')

    # Retornar "Perfect" se não houver erros encontrados
    return "You can proceed"


@app.callback(
    [Output('hidden-div', 'data'), Output("error-message", "children")],
    [Input('validate-button', 'n_clicks')],
    [State({'type': 'dynamic-checklist', 'index': ALL}, 'value')],
    State('criteria-names', 'data'),
    State('num-criteria', 'value')
)
def update_hidden_div_and_validate(n_clicks, values, criteria_names, num_criteria):
    if n_clicks is not None and n_clicks > 0 and criteria_names is not None:
        if values is None:
            values = []

        options = []
        preferences_dict = {}

        importance_selections = {}
        scale_selections = {}
        equal_selections = set()

        for value_list in values:
            if value_list is None:
                continue

            for val in value_list:
                parts = val.split('-')
                i, j = map(int, parts[:2])
                k = parts[2]

                if k == 'i':
                    importance_selections[(i, j)] = i
                elif k == 'j':
                    importance_selections[(i, j)] = j
                elif k == 'equal':
                    equal_selections.add((i, j))
                else:
                    scale_selections[(i, j)] = int(k)

        for (i, j) in importance_selections.keys():
            if (i, j) in equal_selections:
                options.append({'label': f'{criteria_names[i]} and {criteria_names[j]} have equal importance',
                                'value': f'{i}-{j}-equal'})
                preferences_dict[(min(i, j), max(i, j))] = 1
            else:
                selected_criterion = importance_selections[(i, j)]
                importance_factor = scale_selections.get((i, j), 1)
                if selected_criterion == i:
                    options.append({'label': f'{criteria_names[i]} is more important than {criteria_names[j]} by {importance_factor}',
                                    'value': f'{i}-{j}-{importance_factor}'})
                    preferences_dict[(i, j)] = importance_factor
                else:
                    options.append({'label': f'{criteria_names[j]} is more important than {criteria_names[i]} by {importance_factor}',
                                    'value': f'{j}-{i}-{importance_factor}'})
                    preferences_dict[(j, i)] = importance_factor

        for (i, j) in equal_selections:
            if (i, j) not in importance_selections:
                options.append({'label': f'{criteria_names[i]} and {criteria_names[j]} have equal importance',
                                'value': f'{i}-{j}-equal'})
                preferences_dict[(min(i, j), max(i, j))] = 1
        for option in options:
            if 'equal' in option['value']:
                option['value'] = option['value'].replace('equal', '1')
        print(options)
        
        print(preferences_dict)
        error_message = validate_preferences(criteria_names, preferences_dict, num_criteria)
        return options, error_message
    return None, None


@app.callback(
    [Output('plot', 'style'), Output('plot', 'figure')],
    [Input('see-preferences-button', 'n_clicks')],
    [State('criteria-names', 'data'), State('alternatives-names', 'data'), State('hidden-div', 'data')]
)
def show_plot(n, criteria_names, alternatives_names, preferences):
    if n is None or n == 0:
        return {'display': 'none'}, go.Figure()
    else:
        # Generate color scales
        criteria_colors = generate_color_scale('blue', len(criteria_names))

        # Create the bar plot
        fig = go.Figure()
        added_criteria = []
        for preference in preferences:
            i, j, k = map(int, preference['value'].split('-'))  # Use 'value' key to access the preference value
            if k != 1:  # Only add the bar if the preference value is not equal to 1
                if criteria_names[i] not in added_criteria:
                    fig.add_trace(go.Bar(name=f'C{i+1}', x=[f'C{i+1}>C{j+1}'], y=[k], marker_color=criteria_colors[i], width=0.3))
                    added_criteria.append(criteria_names[i])
                else:
                    fig.add_trace(go.Bar(name=f'C{i+1}', x=[f'C{i+1}>C{j+1}'], y=[k], marker_color=criteria_colors[i], showlegend=False, width=0.3))

        # Update the layout to group the bars by preference pairs and set the background to white
        fig.update_layout(xaxis_title={'text':'Preferences Made', 'font': {'size': 12}},
            yaxis_title={'text':'Value Assigned','font': dict(size=12)},barmode='group', bargap=0, plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="left", x=0), autosize=True, height=300, 
                          title={
                'text': 'Are you happy with your preferences?',
                'y':0.98,
                'x':0.3,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=12)
             },margin=dict(t=30, l=10))

        # Update the y-axis to have fixed range
        fig.update_yaxes(range=[0, 9])

        return {'display': 'block', 'width': '100%'}, fig




@app.callback(
    [Output('comparison-matrix', 'data'),
     Output('comparison-matrix', 'columns'),
     Output('max-eigenvalue', 'children'),
     Output('principal-eigenvector', 'children'),
     Output('consistency-ratio', 'children')],
    Input('generate-button', 'n_clicks'),
    State('hidden-div', 'data'),
    State('num-criteria', 'value')
)
def update_outputs(n_clicks, preferences, num_criteria):
    if n_clicks == 0 or preferences is None:
        return [], [], "", "", ""

    preferences_dict = {(int(p['value'].split('-')[0]), int(p['value'].split('-')[1])): float(p['value'].split('-')[2]) for p in preferences}
    weights,v, matrix = calculate_weights_and_v(num_criteria, preferences_dict)
    
    # Calculate the largest eigenvalue (lambda_max)
    eigenvalues, _ = np.linalg.eig(matrix)
    lambda_max = np.real(np.max(eigenvalues))
    lambda_max= np.round(lambda_max,2)
    print(weights)
    weights = np.round(weights,2)
    principal_eigenvector = pd.DataFrame(weights)
    

    principal_eigenvector_str = "\n".join([str(val[0]) for val in principal_eigenvector.values])



    



    
    # Calculate the consistency ratio
    CR = calculate_consistency_ratio(matrix, num_criteria)
    CR = np.round(CR,2)
    matrix = np.round(matrix, 2)
    # Convert the matrix to a list of dictionaries
    matrix_df = pd.DataFrame(matrix)
    matrix_list = matrix_df.to_dict('records')

    # Create a list of dictionaries for the columns property
    columns = [{"name": str(i), "id": str(i)} for i in matrix_df.columns]

    return matrix_list, columns, str(lambda_max), principal_eigenvector_str, str(CR)





@app.callback(
    [Output('consistency-ratio-graph', 'figure'),  # Change 'children' back to 'figure'
     Output('weights-store', 'data'),
     Output('matrix-store', 'data'),
     Output('consistency-ratios-store', 'data'),
     Output('comparison-matrices-store', 'data')],
    Input('generate-button', 'n_clicks'),
    State('hidden-div', 'data'),  # Replace 'preferences-dropdown' with 'hidden-div'
    State('num-criteria', 'value')
)
def update_graph_and_store(n_clicks, preferences, num_criteria):  # 'preferences' now refers to the data in 'hidden-div'
    if n_clicks == 0 or preferences is None:
        return go.Figure(), [], [], [], []

    preferences_dict = {(int(p['value'].split('-')[0]), int(p['value'].split('-')[1])): float(p['value'].split('-')[2]) for p in preferences}
    print(preferences_dict)
    
    print(num_criteria)  # Use 'value' key to access the preference value
    weights,v, matrix = calculate_weights_and_v(num_criteria, preferences_dict)
    print(matrix)
    weights, matrices, consistency_ratios = adjust_preferences(num_criteria, preferences_dict, weights, matrix)

    figure = go.Figure(data=[
                    go.Bar(x=list(range(len(consistency_ratios))), y=consistency_ratios, marker_color='rgb(138, 208, 235)')
                ], layout=go.Layout(
                    title=go.layout.Title(
                        text='Consistency Ratio per Iteration',
                        font=dict(
                            family="serif",
                            size=20,
                            color="#656565",
                            
                        ),
                        x=0
                    ),
                    xaxis_title='Iteration',
                    yaxis_title='Consistency Ratio',
                    legend={'xanchor': 'left'},
                    plot_bgcolor='white'
                ))

    matrices_list = [m.tolist() for m in matrices]
    return figure, weights, matrix, consistency_ratios, matrices_list
        # Return the figure

@app.callback(
    [Output('selected-bars-store', 'data'),
     Output('clicked-heatmap-indices-store', 'data')],
    Input('consistency-ratio-graph', 'clickData'),
    [State('selected-bars-store', 'data'),
     State('clicked-heatmap-indices-store', 'data')]
)
def update_selected_bars_and_clicked_heatmap_indices(clickData, selected_bars, clicked_heatmap_indices):
    if clickData is None:
        return dash.no_update

    new_selected_bar = clickData['points'][0]['x']

    if new_selected_bar not in selected_bars:
        selected_bars.append(new_selected_bar)
    if new_selected_bar not in clicked_heatmap_indices:
        clicked_heatmap_indices.append(new_selected_bar)

    return selected_bars, clicked_heatmap_indices

@app.callback(
    Output('units-container', 'children'),
    [Input('selected-bars-store', 'data'),
     Input('clicked-heatmap-indices-store', 'data')],
    [State('comparison-matrices-store', 'data'),
     State('weights-store', 'data'),
     State('criteria-names', 'data')]
)
def update_units_container(selected_bars, clicked_heatmap_indices, matrices, weights, criteria_names):
    if selected_bars is None or matrices is None or weights is None or clicked_heatmap_indices is None:
        return []

    # Limit the number of selected bars to the last two
    if len(selected_bars) > 2:
        selected_bars = selected_bars[-2:]

    units = []
    for i in selected_bars:
        if i in clicked_heatmap_indices:
            matrix_list = [[round(value, 2) for value in row] for row in matrices[i]]

        

            # Create a HTML table from the matrix with cell borders
            matrix_html = html.Table([
                html.Tr([html.Td(value, style={'border': '1px solid black', 'text-align': 'center', 'padding': '3px'}) for value in row]) for row in matrix_list  # This line adds cell borders and aligns the text to the center
            ], style={'border-collapse': 'collapse', 'margin': '0 auto', 'text-align': 'center'}, className='suggestion')  #
            radar_chart = dcc.Graph(figure=go.Figure(data=[go.Scatterpolar(r=weights[i], theta=[f'C{j+i}' for j in range(len(weights[i]))], fill='toself', marker_color='rgb(138, 208, 235)')],
                                             layout=go.Layout(title=f"Weights for each criterion (Iteration {i})", font=dict(
            family="serif",
            size=14,
            color="#656565"
        ), title_x=0, title_y=0.98, height=450, width=450, margin=dict(t=15),polar_bgcolor='rgb(248, 244, 244)')))

            unit = dbc.Col([
                html.Br(),
                html.Div(f'Matrix of preferences (Iteration {i})', style={'textAlign': 'left', 'fontSize': 20}, className='suggestion'),
                html.Br(),
                matrix_html,  # This line displays the matrix as preformatted text  # This line displays the matrix as preformatted text
                html.Br(),
                radar_chart,
                html.Button('Choose Iteration', id={'type': 'choose-iteration-button', 'index': i}, n_clicks=0, className='btn btn-outline-primary custom-button'),
                html.Div(id={'type': 'weights-output', 'index': i}, children=[]),
            ], width=6)

            units.append(unit)

    return dbc.Row(units)


@app.callback(
    Output({'type': 'weights-output', 'index': MATCH}, 'children'),
    Output({'type': 'selected-weights-store', 'index': MATCH}, 'data'),
    Input({'type': 'choose-iteration-button', 'index': MATCH}, 'n_clicks'),
    State('weights-store', 'data'),
    State({'type': 'selected-weights-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def display_selected_weights(n_clicks, weights, selected_weights):
    if n_clicks is None or n_clicks <= 0:
        return dash.no_update, dash.no_update
    # Display the weights of the selected iteration
    selected_weights.append(weights[n_clicks-1])
    return f"You can proceed", selected_weights


@app.callback(
    Output('weights-graph-container', 'children'),
    Input('see-weights-button', 'n_clicks'),
    [State({'type': 'selected-weights-store', 'index': ALL}, 'data'),
     State('criteria-names', 'data')]
)
def update_bar_chart(n_clicks, selected_weights, criteria_names):
    if n_clicks is None or n_clicks <= 0:
        # Return None
        return None
    
    # Flatten the list of selected weights
    selected_weights = [weight for sublist in selected_weights for weight in sublist]
    selected_weights = selected_weights[0]

    # Generate a color scale
    criteria_colors = generate_color_scale('blue', len(criteria_names))

    # Create new labels C1, C2, ..., Cn
    new_labels = [f'C{i+1}' for i in range(len(criteria_names))]

    # Combine new labels, weights and colors into a single list
    combined_list = list(zip(new_labels, selected_weights, criteria_colors))

    # Sort the combined list by the weights in ascending order
    sorted_list = sorted(combined_list, key=lambda x: x[1])

    # Unzip the sorted list back into new labels, weights and colors
    new_labels, selected_weights, criteria_colors = zip(*sorted_list)

    # Create a horizontal bar chart with the selected weights
    figure = go.Figure(data=[go.Bar(x=selected_weights, y=new_labels, orientation='h', marker_color=criteria_colors, width=0.5)])
    
    figure.update_layout(
        xaxis_title={
                'text': 'Weights',
                
                'font': dict(size=12)
            },
        yaxis_title={
                'text': 'Criteria',
                
                'font': dict(size=12)
            },
        title={
                'text': 'How important is each criterion?',
                'y':0.98,
                'x':0.3,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=12)
             }, font=dict(size=12),
        autosize=True,  # Disable autosize to align the chart to the left
        #width=300,  # Adjust the width as needed
        height=350,
        margin=dict(t=30, l=10),  # Adjust the height as needed
        plot_bgcolor='white',  # Set the background color to white
    )
    # Return a new dcc.Graph component
    return dcc.Graph(figure=figure)



#TOPSIS

@app.callback(
Output('distance-result', 'children'),
Input('hide2-button', 'n_clicks'),
State('values-matrix', 'data'),
State('criteria-types', 'data'),
State({'type': 'selected-weights-store', 'index': ALL}, 'data'),
State('alternatives-names', 'data'),
State('criteria-names', 'data')
)
def show_distances(n_clicks, values_matrix, criteria_types, weights, alternatives_names, criteria_names):
    if n_clicks > 0 and values_matrix and criteria_types and weights:
        weighted_matrix, ideal_solution, neg_ideal_solution, _, _, _ = perform_topsis(values_matrix, criteria_types, weights)


    # Create new labels A1, A2, ..., An for alternatives
        new_alt_labels = [f'A{i+1}' for i in range(len(alternatives_names))]
        alternatives_colors = generate_color_scale('red', len(new_alt_labels))

        # Create new labels C1, C2, ..., Cn for criteria
        new_crit_labels = [f'C{i+1}' for i in range(len(criteria_names))]

        df = pd.DataFrame(weighted_matrix.T, columns=new_alt_labels)

        # Create a line chart
        fig = go.Figure()
        for i, col in enumerate(df.columns):
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[col], 
                mode='lines', 
                name=col,
                line=dict(color=alternatives_colors[i], width=2)  # Increase line width
            ))

        # Add lines for the ideal and negative ideal solutions
        fig.add_trace(go.Scatter(x=df.index, y=ideal_solution, mode='lines', name='Ideal Solution', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=neg_ideal_solution, mode='lines', name='Negative Ideal Solution', line=dict(color='red', dash='dash')))

        # Update layout
        fig.update_layout(
            title='How close are the alternatives?',
            xaxis=dict(
                title='Criteria', 
                title_font=dict(size=10),
                tickvals=list(range(len(new_crit_labels))),  # Set tick values to the indices of the criteria
                ticktext=new_crit_labels  # Set tick text to the new labels of the criteria
            ),
            yaxis=dict(title='Weighted values', title_font=dict(size=10)),
            plot_bgcolor='white',  # Set the background color to white
            title_font_size=12,  # Set the font size of the title to 12
            margin=dict(t=30, l=50),  # Set the margin of the plot
            legend=dict(orientation="h", yanchor="bottom", y=-.52, xanchor="left", x=0)  # Move the legend to below the plot
        )
        return dcc.Graph(figure=fig)
            




@app.callback(
    Output('performance-result', 'children'),
    Input('see-performance-button', 'n_clicks'),
    State('values-matrix', 'data'),
    State('criteria-types', 'data'),
    State({'type': 'selected-weights-store', 'index': ALL}, 'data'),
    State('alternatives-names', 'data'),
    State('criteria-names', 'data')
)
def show_performance(n_clicks, values_matrix, criteria_types, weights, alternatives_names, criteria_names):
    if n_clicks is not None and n_clicks > 0 and values_matrix and criteria_types and weights:
        weighted_matrix, _, _, _, _, _ = perform_topsis(values_matrix, criteria_types, weights)
        alternatives_colors = generate_color_scale('red', len(alternatives_names))
        weighted_matrix = abs(weighted_matrix)

        # Create new labels A1, A2, ..., An for alternatives
        new_alt_labels = [f'A{i+1}' for i in range(len(alternatives_names))]
        
        # Create new labels C1, C2, ..., Cn for criteria
        new_crit_labels = [f'C{i+1}' for i in range(len(criteria_names))]

        # Create a DataFrame from the weighted matrix
        df = pd.DataFrame(weighted_matrix.T, columns=new_alt_labels)
        
        # Create a multiple trace radar chart
        fig = go.Figure()
        for i, alt in enumerate(df.columns):
            fig.add_trace(go.Scatterpolar(
                r=df[alt], 
                theta=new_crit_labels, 
                fill='toself', 
                name=alt,
                line=dict(color=alternatives_colors[i], width=0),
                fillcolor=alternatives_colors[i],
                opacity=0.65,  # Adjust the opacity
            ))
        
        # Update layout
        fig.update_layout(
            title='Any with more impact?',
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=True,),
                bgcolor='white',
            ),
            showlegend=True,
            legend=dict(
                title=dict(text='Alternatives', font=dict(size=12)),
                orientation='h',
                yanchor='bottom',
                y=-.25,
                xanchor='left',
                x=0
            ),
            polar_bgcolor='rgb(248, 244, 244)',  # Set the background color of the entire chart area to white
            title_font_size=12,  # Set the font size of the title to 12
            margin=dict(t=45, l=85),
            autosize=False,  # Disable autosize to manually set the size
            width=400,  # Set the width of the plot
            height=400,
        )
                
        return dcc.Graph(figure=fig)


@app.callback(
    Output('ranking-result', 'children'),
    Input('ranking-button', 'n_clicks'),
    State('values-matrix', 'data'),
    State('criteria-types', 'data'),
    State({'type': 'selected-weights-store', 'index': ALL}, 'data'),
    State('alternatives-names', 'data')
)
def show_ranking(n_clicks, values_matrix, criteria_types, weights, alternatives_names):
    if n_clicks > 0 and values_matrix and criteria_types and weights:
        print("entrou no ranking")
        weighted_matrix, _, _, sep_measures, neg_sep_measures, _ = perform_topsis(values_matrix, criteria_types, weights)
        closeness = neg_sep_measures / (sep_measures + neg_sep_measures)
        sep_measures = abs(sep_measures)
        neg_sep_measures = abs(neg_sep_measures)  # Use the absolute value of the closeness

        # Create new labels A1, A2, ..., An
        new_labels = [f'A{i+1}' for i in range(len(alternatives_names))]
        alternatives_colors = generate_color_scale('red', len(new_labels))

        print(closeness)
        print(sep_measures)
        print(neg_sep_measures)

        # Create a ternary scatter plot
        fig = go.Figure()
        max_closeness = max(closeness)
        max_sep = max(sep_measures)
        max_neg = max(neg_sep_measures)  # Get the maximum closeness value

        for i, alt in enumerate(new_labels):
            normalized_closeness = closeness[i] / max_closeness
            normalized_sep = sep_measures[i] / max_sep
            normalized_neg = neg_sep_measures[i] / max_neg
            print("normalized closeness", normalized_closeness)
            print("normalized sep", normalized_sep)  # Normalize the closeness value
            print("normalized neg", normalized_neg)
            fig.add_trace(go.Scatterternary(
                a=[normalized_closeness],
                b=[normalized_sep],
                c=[normalized_neg],
                mode='markers',
                marker=dict(size=20, color=alternatives_colors[i]),  # Increase the size to 15
                name=alt
            ))

        # Update layout
        fig.update_layout(
            title='Which is the winner?',
            ternary=dict(
                bgcolor='rgb(248, 244, 244)',
                sum=1,
                aaxis=dict(showticklabels=False, title='Ranking', title_font_size=9),  # Hide labels on a axis
                baxis=dict(showticklabels=False, title='  Closeness to the Ideal', title_font_size=9),  # Hide labels on b axis
                caxis=dict(showticklabels=False, title='Closeness to the Worst', title_font_size=9),  # Hide labels on c axis
            ),
            autosize=False,
            width=400,
            height=400,
            margin=dict(t=30, l=60, b=50, r=50, pad=5),
            paper_bgcolor='white',
            title_font_size=12,  # Set the font size of the title to 12
        )

        return dcc.Graph(figure=fig)


@app.callback(
    Output('cytoscape', 'elements'),
    Input('confirm-button', 'n_clicks'),
    State('criteria-names', 'data'),
    State('hidden-div', 'data'),
    State('situation-to-solve', 'value')
)
def update_graph(n_clicks, criteria_names, preferences, situation_to_solve):
    if n_clicks is not None and n_clicks > 0 and criteria_names is not None:
        # Create a dictionary to store the sum of preferences for each criterion
        criteria_preferences = {name: 0 for name in criteria_names}

        # Update the sum of preferences for each criterion
        for preference in preferences:
            i, j, k = map(int, preference['value'].split('-'))
            criteria_preferences[criteria_names[i]] += k 
            criteria_preferences[criteria_names[j]] += 1/k 

        criteria_colors = generate_color_scale('blue', len(criteria_names))
        

        # Create the nodes and edges for the graph
        nodes = [{'data': {'id': 'problem', 'label': situation_to_solve}, 'style': {'background-color': 'rgb(81, 80, 108)', 'font-family': 'Poppins'}}] + [{'data': {'id': name, 'label': name}, 'style': {'width': 2*math.sqrt(criteria_preferences[name]*100/math.pi), 'height': 2*math.sqrt(criteria_preferences[name]*100/math.pi), 'background-color': criteria_colors[i], 'text-valign': 'bottom', 'font-family': 'Poppins'}} for i, name in enumerate(criteria_names)]
        edges = [{'data': {'source': 'problem', 'target': name}, 'style': {'line-color': criteria_colors[criteria_names.index(name)]}} for name in criteria_names]
        elements = nodes + edges
    else:
        elements = []
    return elements






@app.callback(
    [Output('graph-container', 'children'), Output('values-matrix', 'data')],
    Input('save-values-button', 'n_clicks'),
    [State({'type': 'value', 'index': ALL}, 'value'),
     State('num-criteria', 'value'),
     State('num-alternatives', 'value'),
     State('criteria-names', 'data'),
     State('alternatives-names', 'data'),
     State('situation-to-solve', 'value'),
     State('hidden-div', 'data')]
)
def update_graph_and_save_values(n_clicks, values, num_criteria, num_alternatives, criteria_names, alternatives_names, situation_to_solve, preferences):
    if n_clicks is not None and n_clicks > 0 and criteria_names is not None:
        # Save the values of the matrix
        matrix = np.array(values).reshape((num_alternatives, num_criteria))
        saved_values_matrix = matrix.tolist()
        
        print (criteria_names)
        print (alternatives_names)


        criteria_preferences = {name: 0 for name in criteria_names}

        # Update the sum of preferences for each criterion
        for preference in preferences:
            i, j, k = map(int, preference['value'].split('-'))
            criteria_preferences[criteria_names[i]] += k 
            criteria_preferences[criteria_names[j]] += 1/k 

        scaler = MinMaxScaler()

        # Fit the scaler to the data and transform the data
        normalized_values_matrix = scaler.fit_transform(saved_values_matrix)
        normalized_values_matrix = [[item + 10 for item in sublist] for sublist in normalized_values_matrix]
        print(normalized_values_matrix)
            

        labels = [situation_to_solve] + criteria_names + [f"{a}_{c}" for c in criteria_names for a in alternatives_names]
        parents = [""] + [situation_to_solve]*len(criteria_names) + [c for c in criteria_names for a in alternatives_names ]
        values = [0] + [criteria_preferences[c] for c in criteria_names] + [normalized_values_matrix[alternatives_names.index(a)][criteria_names.index(c)] for c in criteria_names for a in alternatives_names]
        criteria_colors = generate_color_scale('blue', len(criteria_names))
        alternatives_colors = generate_color_scale('red', len(alternatives_names))

        # Create a color list that matches the labels list
        colors = [""] + criteria_colors + [alternatives_colors[alternatives_names.index(a)] for c in criteria_names for a in alternatives_names]
        
        text = [situation_to_solve] + criteria_names + [f"{a}: {saved_values_matrix[alternatives_names.index(a)][criteria_names.index(c)]}" for c in criteria_names for a in alternatives_names]
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            text=text,
            textinfo="label+text",
            marker_colors=colors

        ))
        fig.update_layout(
                #title='Summary of your problem',
                margin=dict(t=30, l=50, b=50, r=50, pad=5),
                paper_bgcolor='white',  # Set the background color of the entire chart area to white
                #title_font_size=12,  # Set the font size of the title to 12
            )

        return dcc.Graph(figure=fig), saved_values_matrix
    else:
        return [], None





@app.callback(
    Output('values-matrix-table', 'children'),
    Input('see-values-matrix-button', 'n_clicks'),
    State('values-matrix', 'data'),
    State('criteria-types', 'data'),
    State({'type': 'selected-weights-store', 'index': ALL}, 'data'),
    State('alternatives-names', 'data'),
    State('criteria-names', 'data')
)
def show_values_matrix(n_clicks, values_matrix, criteria_types, weights, alternatives_names, criteria_names):
    if n_clicks is not None and n_clicks > 0 and values_matrix and criteria_types and weights:
        # Create a DataFrame from the values matrix
        df = pd.DataFrame(values_matrix, columns=criteria_names, index=alternatives_names)
        df = df.apply(pd.to_numeric, errors='coerce')
        # Create a list for the style_data_conditional property
        style_data_conditional = []
        for i in range(len(df.columns)):
            if criteria_types[i] == 'benefit':
                max_value = df[df.columns[i]].max()
                min_value = df[df.columns[i]].min()
                style_data_conditional.extend([
                    {'if': {'column_id': df.columns[i], 'filter_query': '{{{}}} = {}'.format(df.columns[i], max_value)},
                     'backgroundColor': 'rgb(7, 205, 33)'},
                    {'if': {'column_id': df.columns[i], 'filter_query': '{{{}}} = {}'.format(df.columns[i], min_value)},
                     'backgroundColor': 'rgb(243, 31, 31)'}
                ])
            else:  # criteria type is 'cost'
                max_value = df[df.columns[i]].max()
                min_value = df[df.columns[i]].min()
                style_data_conditional.extend([
                    {'if': {'column_id': df.columns[i], 'filter_query': '{{{}}} = {}'.format(df.columns[i], min_value)},
                     'backgroundColor': 'rgb(7, 205, 33)'},
                    {'if': {'column_id': df.columns[i], 'filter_query': '{{{}}} = {}'.format(df.columns[i], max_value)},
                     'backgroundColor': 'rgb(243, 31, 31)'}
                ])

        style_data_conditional.append(
            {
                'if': {'column_id': df.reset_index().columns[0]},
                'backgroundColor': 'rgb(255, 255, 255)',
                'color': '#656565',  # Text color
                'font-size': '16px',  # Font size
                'font-family': 'serif',  # Font family
                'font-weight': 'normal'
            }
        )
        
        # Convert the DataFrame to a Dash DataTable
        df.index.name = "Alt/Cri"
        table = dash_table.DataTable(
            data=df.reset_index().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.reset_index().columns],
            style_cell_conditional=[
                {'if': {'column_id': c},
                 'textAlign': 'left'} for c in df.columns],
            style_data_conditional=style_data_conditional,
            style_cell={'width': '30px', 'minWidth': '30px', 'maxWidth': '30px', 'whiteSpace': 'normal'},
            
            style_header={
                'backgroundColor': 'rgb(255, 255, 255)',
                'color': '#656565',  # Text color
                'font-size': '16px',  # Font size
                'font-family': 'serif',  # Font family
                'font-weight': 'normal'
            }
        )
        
        return table


@app.callback(
    [Output('intro', 'style'),
     Output('problem-structuring', 'style'),
     Output('criteriabox', 'style'),
     Output('criteriabox2', 'style'),
     Output('criteriafinal', 'style'),
     Output('alternativebox', 'style'),
     Output('insertvalues', 'style'),
     Output('alternativeboxsure', 'style'),
     # add more outputs as needed...
     Output('current-div', 'data')],
    [Input('back-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('current-div', 'data')]
)
def navigate(n_back, n_next, current_div):
    div_order = ['intro', 'problem-structuring' ,'criteriabox', 'criteriabox2', 'criteriafinal', 'alternativebox','insertvalues', 'alternativeboxsure']  # specify the order of divs
    current_index = div_order.index(current_div)

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'next-button' and current_index < len(div_order) - 1:
            current_index += 1
        elif button_id == 'back-button' and current_index > 0:
            current_index -= 1

    styles = [{'display': 'block' if div == div_order[current_index] else 'none'} for div in div_order]
    return styles + [div_order[current_index]]



@app.callback(
    [Output('explanation-box', 'style'),
     Output('ahpbox', 'style'),
     Output('consistency', 'style'),
     Output('consistency-choice', 'style'),
     
     # add more outputs as needed...
     Output('current-div2', 'data')],
    [Input('back-button2', 'n_clicks'),
     Input('next-button2', 'n_clicks')],
    [State('current-div2', 'data')]
)
def navigate(n_back, n_next, current_div):
    div_order = ['explanation-box', 'ahpbox' ,'consistency', 'consistency-choice']  # specify the order of divs
    current_index = div_order.index(current_div)

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'next-button2' and current_index < len(div_order) - 1:
            current_index += 1
        elif button_id == 'back-button2' and current_index > 0:
            current_index -= 1

    styles = [{'display': 'block' if div == div_order[current_index] else 'none'} for div in div_order]
    return styles + [div_order[current_index]]



@app.callback(
    [Output('topsis', 'style'),
     Output('topsis2box', 'style'),
     Output('topsisconclusion', 'style'),
     
     
     # add more outputs as needed...
     Output('current-div3', 'data')],
    [Input('back-button3', 'n_clicks'),
     Input('next-button3', 'n_clicks')],
    [State('current-div3', 'data')]
)
def navigate(n_back, n_next, current_div):
    div_order = ['topsis', 'topsis2box' ,'topsisconclusion']  # specify the order of divs
    current_index = div_order.index(current_div)

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'next-button3' and current_index < len(div_order) - 1:
            current_index += 1
        elif button_id == 'back-button3' and current_index > 0:
            current_index -= 1

    styles = [{'display': 'block' if div == div_order[current_index] else 'none'} for div in div_order]
    return styles + [div_order[current_index]]


if __name__ == '__main__':
    app.run_server(debug=False)


