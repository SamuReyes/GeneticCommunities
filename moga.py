from concurrent.futures import ProcessPoolExecutor
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import cdlib
import math

class UnionFind:
    """ Estructura auxiliar para decodificar un individuo en modo locus a modo cluster """
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1

class MOGA():
        
    def __init__(self, graph, N=100, init=0.5, pcross=0.7, pmut=0.1, n_iter=1000, fitness_metrics=1, n_tour = 4, crossover_op=2):
        self.graph = graph # Grafo
        self.N = N # Tamaño de la población
        self.pop = [] # Población
        self.init = init # Ponderación de la inicialización
        self.pcross = pcross # Probabilidad de cruce
        self.pmut = pmut # Probabilidad de mutación
        self.n_iter = n_iter # Numero de iteraciones
        self.fitness_metrics = fitness_metrics # Par de métricas utilizadas para el fitness
        self.n_tour = n_tour # Número de participantes en el torneo
        self.crossover_op = crossover_op # Operador de cruce

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
	
    def random_init(self) -> list[int]:
        """ Inicializa individuo de forma aleatoria """

        graph_label = [-1] * len(self.graph.nodes())
        for node in list(self.graph.nodes()):
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) == 0:
                graph_label[node] = node
            else:
                graph_label[node] = random.choice(neighbors)

        return graph_label

    def label_propagation_init(self) -> list[int]:
        """ Inicializa un individuo mediante el algoritmo de propagación de etiquetas """

        # Se obtienen las comunidades mediante el algoritmo de propagación de etiquetas
        community_dict_values = nx.algorithms.community.asyn_lpa_communities(self.graph)

        # Se crea un mapping de nodos a comunidades
        node_community_map = {}
        for community_id, nodes in enumerate(community_dict_values):
            for node in nodes:
                node_community_map[node] = community_id

        locus_representation = [-1] * len(self.graph.nodes()) 
        
        # Para cada nodo se asigna un nodo vecino de la misma comunidad
        for node in self.graph.nodes():
            community = node_community_map[node] 
            same_community_neighbors = [neighbor for neighbor in self.graph.neighbors(node) if node_community_map[neighbor] == community]
            if same_community_neighbors:
                locus_representation[node] = random.choice(same_community_neighbors)
            else:
                locus_representation[node] = node

        return locus_representation

    def create_pop(self):
        """ Crea una población de N individuos codificados en modo locus """

        random_pop = math.ceil(self.N*self.init)
        label_propagation_pop = self.N - random_pop
        
        # Se inicializa la población con individuos aleatorios
        for _ in range(random_pop):
            self.pop.append(self.random_init())
            
        # Se inicializa el resto de la población mediante el algoritmo de propagación de etiquetas
        for _ in range(label_propagation_pop):
            self.pop.append(self.label_propagation_init())

    def decode(self, locus_representation: list[int]) -> dict[int, int]:
        """ Decodifica un individuo en modo locus a modo cluster """
        uf = UnionFind(len(locus_representation))

        for node, neighbor in enumerate(locus_representation):
            uf.union(node, neighbor)

        # Agrupar nodos por su raíz en la estructura Union-Find
        communities = {}
        for node in range(len(locus_representation)):
            root = uf.find(node)
            if root not in communities:
                communities[root] = []
            communities[root].append(node)

        communities = list(communities.values())

        return communities
    
    def plot(self, node_community_map: list[list[int]]):
        """ Pinta el grafo con un color por comunidad """

        # Se crea un mapping de nodos a comunidades
        node_color_map = {}
        for idx, community in enumerate(node_community_map):
            for node in community:
                node_color_map[node] = idx

        # Se obtiene la lista de colores        
        color_map = []
        for node in self.graph.nodes():
            color_map.append(node_color_map[node])

        plt.figure(figsize=(30, 25))
        nx.draw(self.graph, node_color=color_map, with_labels=True)      
        plt.show()

    def plot_pareto_front(self, first_front, fitness_values):
        """ Pinta el frente de Pareto """
        # Extraer los valores de fitness para los individuos en el primer frente
        x_values = [fitness_values[i][0] for i in first_front]
        y_values = [fitness_values[i][1] for i in first_front]

        # Crear el plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, c='blue', marker='o')
        plt.title("Frente de Pareto")
        plt.xlabel("Objetivo 1")
        plt.ylabel("Objetivo 2")
        plt.grid(True)
        plt.show()
    
    def fitness(self, individual: list[int]) -> float:
        """ Calcula el fitness de un individuo

        0: {Community score+, Internal density}
        1: {Q +, Internal density +}
        2: {Community score+, Average-ODF}
        3: {Community score+, Max-ODF}

        """
        
        communities = self.decode(individual)

        NodeClustering = cdlib.NodeClustering(communities, self.graph)
        NodeClusteringCommunities = [cdlib.NodeClustering([community], self.graph) for community in communities]

        if self.fitness_metrics == 0:
            community_score = sum(cdlib.evaluation.average_internal_degree(self.graph, NodeClustering, summary=False))
            internal_density = sum(1 - cdlib.evaluation.internal_edge_density(self.graph, community).score for community in NodeClusteringCommunities)
            return [community_score, internal_density]
        
        elif self.fitness_metrics == 1:
            q = cdlib.evaluation.newman_girvan_modularity(self.graph, NodeClustering).score
            internal_density = sum(1 - cdlib.evaluation.internal_edge_density(self.graph, community).score for community in NodeClusteringCommunities)
            return [q, internal_density]
        
        elif self.fitness_metrics == 2:
            avg_odf = sum(cdlib.evaluation.avg_odf(self.graph, community).score for community in NodeClusteringCommunities)
            community_score = sum(cdlib.evaluation.average_internal_degree(self.graph, NodeClustering, summary=False))
            return [community_score, avg_odf]
        
        elif self.fitness_metrics == 3:
            max_odf = sum(cdlib.evaluation.max_odf(self.graph, community).score for community in NodeClusteringCommunities)
            community_score = sum(cdlib.evaluation.average_internal_degree(self.graph, NodeClustering, summary=False))
            return (community_score, max_odf)
        
    def dominates(self, individual1:float, individual2:float) -> bool:
        """ Devuelve True si individual1 domina a individual2 """
        return all(x >= y for x, y in zip(individual1, individual2)) and any(x > y for x, y in zip(individual1, individual2))
    

    def moga_fast_non_dominated_sort(self, fitness: list[float]) -> list[list[int]]:
        """ Returns a list of Pareto fronts, including empty levels of dominance """
        domination_counts = [0] * len(self.pop)
        dominated_solutions = [set() for _ in self.pop]
        max_dominance = 0

        # Identifying the dominance relationships
        for p in range(len(fitness)):
            for q in range(len(fitness)):
                if p != q:
                    if self.dominates(fitness[p], fitness[q]):
                        dominated_solutions[p].add(q)
                    elif self.dominates(fitness[q], fitness[p]):
                        domination_counts[p] += 1

            max_dominance = max(max_dominance, domination_counts[p])

        # Initializing the fronts list with empty lists
        fronts = [[] for _ in range(max_dominance + 2)]

        # Assigning solutions to the appropriate front
        for idx, count in enumerate(domination_counts):
            # fronts[count].append(self.pop[idx]) #Returns the graph of the individual "idx"
            fronts[count].append(idx)           #Returns the index of the individual "idx"

        return fronts[:-1]
    
    
    def sharing_function(self, distance, sigma = 0.1):
        if distance < sigma:
            return 1 - (distance / sigma)
        else:
            return 0


    def calculate_distance(self, individual1, individual2, fitness):
        distance_pow_2 = (fitness[individual1][0] - fitness[individual2][0])**2 + (fitness[individual1][1] - fitness[individual2][1])**2
        distance = distance_pow_2**0.5
        return distance


    def adjusted_fitness(self, individual, population, sigma, F, fitness):
        shared_fitness = 0
        for other_individual in population:
            if individual != other_individual:
                distance = self.calculate_distance(individual, other_individual, fitness)
                shared_fitness += self.sharing_function(distance, sigma)
                
        shared_fitness = max(shared_fitness, 1)
        return F[individual] / shared_fitness

    def tournament(self, old_fitness):
        """ Selección de padres por torneo """
        participants = random.sample(self.pop, self.n_tour)
        best = None
        for participant in participants:
            if best is None:
                best = participant
            elif self.dominates(old_fitness[self.pop.index(participant)], old_fitness[self.pop.index(best)]):
                best = participant
        return best
    
    def single_point_crossover(self, p1, p2):
        """ Crossover de un punto """
        c1, c2 = p1.copy(), p2.copy()
        
        if np.random.uniform() < self.pcross:
            # seleccionar punto de crossover
            pt = random.randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
    
    def multiple_point_crossover(self, p1, p2, k):
        """ Crossover de k puntos """

        if len(p1) != len(p2):
            raise ValueError("Los padres deben tener el mismo tamaño")

        # Generar k puntos de cruce únicos
        crossover_points = sorted(random.sample(range(1, len(p1)), k))
        offspring1, offspring2 = [], []
        previous_point = 0
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                offspring1.extend(p1[previous_point:point])
                offspring2.extend(p2[previous_point:point])
            else:
                offspring1.extend(p2[previous_point:point])
                offspring2.extend(p1[previous_point:point])
            previous_point = point

        # Agregar el último segmento
        if k % 2 == 0:
            offspring1.extend(p1[previous_point:])
            offspring2.extend(p2[previous_point:])
        else:
            offspring1.extend(p2[previous_point:])
            offspring2.extend(p1[previous_point:])
        return [offspring1, offspring2]

    def uniform_crossover(self, p1,p2):
        """ Crossover uniforme """
        c1, c2 = p1.copy(), p2.copy()
        if np.random.uniform() < self.pcross:
            for i in range(len(p1)):
                if random.randint(0,2) == 1:
                    c1[i] , c2[i] = c2[i] , c1[i]
        return [c1,c2]
    
    def crossover(self, i1, i2):
        """ Cruce de dos individuos """
        if self.crossover_op==0:
            return self.single_point_crossover(i1,i2)
        elif self.crossover_op==1:
            return self.multiple_point_crossover(i1,i2,2)
        elif self.crossover_op==2:
            return self.uniform_crossover(i1,i2)

    def mutate(self, individual: list) -> list[int]:
        """ Mutación de un individuo """
        size = len(individual)
        # Se muta cada nodo con probabilidad pmut cambiando el vecino por uno aleatorio
        for i in range(size):
            if self.__choose_with_prob(self.pmut):
                neighbors = list(self.graph.neighbors(i))
                individual[i] = random.choice(neighbors)

        return individual
    
    def create_children(self, old_fitness):
        """ Crea una población de hijos de tamaño N (tamaño de la población)"""
        children = []
        while len(children) < self.N:
            parent1 = self.tournament(old_fitness)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.tournament(old_fitness)
            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            children.append(child1)
            children.append(child2)

        return children

    def evolve(self):
        """ Evoluciona la población durante n_iter iteraciones """
        
        self.create_pop()
        with ProcessPoolExecutor() as executor:
                old_fitness = list(executor.map(self.fitness, self.pop))

        for _ in range(self.n_iter): 
            
            # Se añaden los hijos a la población
            children = self.create_children(old_fitness)
            self.pop.extend(children)

            # Se calcula el fitness de la población y se calculan los frentes de Pareto
            with ProcessPoolExecutor() as executor:
                new_fitness = list(executor.map(self.fitness, children))
            
            fitness_pop = old_fitness + new_fitness
            
            #Devuelve los rangos de dominancia en una lista de listas.
            # ej: paretos = [[A, B, C, D], [F], [E], [], [], [], [G]] donde paretos[0] (A, B, C, D) representan las soluciones no dominadas
            # paretos[1] (F) representa las soluciones dominadas por una solución y así sucesivamente
            
            paretos = self.moga_fast_non_dominated_sort(fitness_pop)
            
        
        # Se seleccionan los individuos que pasan a la siguiente generación

            # Calculamos el fitness de cada individuo en la población

            # Fi = N - 0.5(μ(ri) - 1) - Σμ(k)
            # N: tamaño de la población
            # μ(ri): rango del individuo i
            # Σμ(k): Nº de individuos con rangos inferiores al individuo i
            
            pop_len = len(fitness_pop)
            F = [0.0] * pop_len
            Σμ_k = 0

            for μ_ri, front in enumerate(paretos):
                
                # Fi = N - 0.5(μ(ri) - 1) - Σμ(k)
                for i_value in front:

                    # F[i_value] = self.N - 0.5 * ((μ_ri+1) - 1) - Σμ_k
                    F[i_value] = pop_len - 0.5 * ((μ_ri+1) - 1) - Σμ_k

                    # We apply a niching technique to keep diversity
                    F[i_value] = self.adjusted_fitness(i_value, front, 0.2, F, fitness_pop)

                Σμ_k += len(front)

            #We sort the indexes of the population by their fitness
            sorted_individuals_by_fitness = sorted(range(len(F)), key=F.__getitem__, reverse=False)
            sorted_individuals_by_fitness

            self.pop = sorted_individuals_by_fitness[:self.N]

            if _ % 100 == 0:
                print(f"Generación {_}")
                self.plot_pareto_front(paretos[0], fitness_pop)
