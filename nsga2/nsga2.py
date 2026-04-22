import numpy as np

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(population, objectives):
    fronts = [[]]
    domination_count = [0]*len(population)
    dominated_solutions = [[] for _ in population]

    for i in range(len(population)):
        for j in range(len(population)):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return fronts[:-1]

def tournament(i, j, rank, crowding):
    if rank[i] < rank[j]:
        return i
    elif rank[i] > rank[j]:
        return j
    else:
        return i if crowding[i] > crowding[j] else j

def crowding_distance(front, objectives):
    distance = [0]*len(front)
    n_obj = len(objectives[0])

    for m in range(n_obj):
        sorted_idx = sorted(range(len(front)), key=lambda i: objectives[front[i]][m])

        distance[sorted_idx[0]] = float('inf')
        distance[sorted_idx[-1]] = float('inf')

        for i in range(1, len(front)-1):
            prev = objectives[front[sorted_idx[i-1]]][m]
            next_ = objectives[front[sorted_idx[i+1]]][m]
            distance[sorted_idx[i]] += (next_ - prev)

    return distance

class NSGA2:
    def __init__(self, pop_size=100, n_pumps=3, mutation_rate=0.3, power_max=1, fiber_len=10):
        self.pop_size = pop_size
        self.n_pumps = n_pumps
        
        self.initial_mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        
        self.mutation_rate = mutation_rate

        self.min_mutation_rate = 0.05
        
        # Limites para os parâmetros
        self.lambda_min = 1360
        self.lambda_max = 1450
        self.power_min = 0.5  # W
        self.power_max = power_max  # W
        self.fiber_len = fiber_len # m

    def initialize_population(self):
        """Inicializa a população com valores aleatórios dentro dos limites."""
        population = []
        # Cada indivíduo é um array concatenado de lambdas e potências

        for _ in range(self.pop_size):
            lambdas = np.sort(np.random.uniform(self.lambda_min, self.lambda_max, self.n_pumps))
            powers = np.random.uniform(self.power_min, self.power_max, self.n_pumps)
            individual = np.concatenate([lambdas, powers])
            population.append(individual)
        
        # good_individual = np.array([1390.51311318, 1403.63761682, 1430.0737674, 2.49978332, 2.49942191, 2.49940851])
        # population[0] = good_individual

        return np.array(population)

    def evaluate_objectives(self, individual, evaluate_amplifier):
    
        lambdas = individual[:self.n_pumps]
        powers = individual[self.n_pumps:]
        
        # Chama a função de avaliação do amplificador
        ripple, gain = evaluate_amplifier(lambdas, powers, self.fiber_len)
        
        # ripple_penalty = 0
        # if ripple > 3:
        #     ripple_penalty = 100 * (ripple - 3) # penalidade suave

        # powers_penalty = 0
        # for p in powers:
        #     if p < self.power_min:
        #         powers_penalty += 100 * (self.power_min - p)
        #     elif p > self.power_max:
        #         powers_penalty += 100 * (p - self.power_max) 
        
        return [ripple, -gain]

    def mutate(self, individual):
        """Aplica mutação gaussiana em um indivíduo."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                if i < self.n_pumps:                    # Mutação nos comprimentos de onda
                    mutated[i] += np.random.normal(0, 1)
                    mutated[i] = np.clip(mutated[i], self.lambda_min, self.lambda_max)
                else:                                   # Mutação nas potências
                    mutated[i] += np.random.normal(0, 0.01)
                    mutated[i] = np.clip(mutated[i], self.power_min, self.power_max)
        
        mutated[:self.n_pumps] = np.sort(mutated[:self.n_pumps])
        return mutated
    
    def crossover(self, parent1, parent2):
        """Realiza o crossover entre dois pais."""
        # Crossover de um ponto
        cross_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
        
        return child1, child2


    def evolve(self, population, evaluate_amplifier, n_generations=3000):
        best_gain = -np.inf
        no_improvement_count = 0
        patience = 300
        tolerance = 0.01

        combined_objectives = None

        history_best_gain = []
        history_best_ripple = []

        for generation in range(n_generations):

            # avaliar população atual
            objectives = np.array([
                self.evaluate_objectives(ind, evaluate_amplifier)
                for ind in population
            ])

            # pareto
            fronts = fast_non_dominated_sort(population, objectives)

            pareto_front = fronts[0]

            # parada precoce
            if generation != 0:
                current_best_gain = max([-combined_objectives[i][0] for i in pareto_front])

                if current_best_gain > best_gain + tolerance:
                    best_gain = current_best_gain
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience and generation > 400:
                    print(f"Parada antecipada na geração {generation+1}")
                    break

            # crowding distance
            crowding = np.zeros(len(population))
            rank = np.zeros(len(population))

            for i, front in enumerate(fronts):
                distances = crowding_distance(front, objectives)
                for j, idx in enumerate(front):
                    crowding[idx] = distances[j]
                    rank[idx] = i

            # seleção
            parents = []
            for _ in range(self.pop_size):
                i, j = np.random.randint(0, self.pop_size, 2)
                winner = i if (
                    (rank[i] < rank[j]) or
                    (rank[i] == rank[j] and crowding[i] > crowding[j])
                ) else j
                parents.append(population[winner])

            parents = np.array(parents)

            # descendentes
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    c1, c2 = self.crossover(parents[i], parents[i+1])
                    offspring.append(self.mutate(c1))
                    offspring.append(self.mutate(c2))
                else:
                    offspring.append(self.mutate(parents[i]))

            offspring = np.array(offspring)

            # elitismo
            combined = np.vstack((population, offspring))

            combined_objectives = np.array([
                self.evaluate_objectives(ind, evaluate_amplifier)
                for ind in combined
            ])

            fronts = fast_non_dominated_sort(combined, combined_objectives)

            # nova população
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    distances = crowding_distance(front, combined_objectives)
                    sorted_front = [x for _, x in sorted(
                        zip(distances, front), reverse=True
                    )]
                    remaining = self.pop_size - len(new_population)
                    new_population.extend(sorted_front[:remaining])
                    break

            population = combined[new_population]

            # pegar melhor da primeira fronteira
            pareto_front = fronts[0]
            best_idx = min(pareto_front, key=lambda i: combined_objectives[i][0])  # menor -gain

            best_ind = combined[best_idx]
            best_ripple, best_gain = evaluate_amplifier(
                best_ind[:self.n_pumps],
                best_ind[self.n_pumps:],
                self.fiber_len
            )

            history_best_gain.append(best_gain)
            history_best_ripple.append(best_ripple)

            if (generation + 1) % 10 == 0:
                print(f"Geração {generation+1}: ganho={best_gain:.2f} dB, ripple={best_ripple:.2f}")

        return population, history_best_gain, history_best_ripple