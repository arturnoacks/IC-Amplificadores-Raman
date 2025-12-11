import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=100, n_pumps=3, mutation_rate=0.3, power_max=1, fiber_len=10):
        self.pop_size = pop_size
        self.n_pumps = n_pumps
        self.mutation_rate = mutation_rate
        
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

    def evaluate_fitness(self, individual, evaluate_amplifier):
    
        lambdas = individual[:self.n_pumps]
        powers = individual[self.n_pumps:]
        
        # Chama a função de avaliação do amplificador
        ripple, gain = evaluate_amplifier(lambdas, powers, self.fiber_len)
        
        ripple_penalty = 0
        if ripple > 3:
            ripple_penalty = 100 * (ripple - 3) # penalidade suave

        powers_penalty = 0
        for p in powers:
            if p < self.power_min:
                powers_penalty += 100 * (self.power_min - p)
            elif p > self.power_max:
                powers_penalty += 100 * (p - self.power_max) 
        
        return gain - ripple_penalty - powers_penalty

    def select_parents(self, population, fitness_scores):
        """Seleciona pais usando uma seleção probabilística."""
        shifted = fitness_scores - fitness_scores.min() + 1e-6
        probs = shifted / shifted.sum()
        idx = np.random.choice(len(population), len(population), p=probs)
        return population[idx]
        

    def crossover(self, parent1, parent2):
        """Realiza o crossover entre dois pais."""
        # Crossover de um ponto (sujeito a mudanças)
        cross_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
        
        return child1, child2

    def mutate(self, individual):
        """Aplica mutação gaussiana em um indivíduo."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                if i < self.n_pumps:                    # Mutação nos comprimentos de onda
                    mutated[i] += np.random.normal(0, 1)
                else:                                   # Mutação nas potências
                    mutated[i] += np.random.normal(0, 0.01)
        
        mutated[:self.n_pumps] = np.sort(mutated[:self.n_pumps])
        return mutated

    def evolve(self, population, evaluate_amplifier, n_generations=100):
        best_fitness = -np.inf
        best_individual = None
        
        for generation in range(n_generations):
            # Avalia o fitness de toda a população
            fitness_scores = np.array([self.evaluate_fitness(ind, evaluate_amplifier) 
                                     for ind in population])
            
            # Atualiza o melhor indivíduo
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx]
            
            parents = self.select_parents(population, fitness_scores)
            
            # Cria nova população através de crossover e mutação
            new_population = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                else:
                    new_population.append(self.mutate(parents[i]))
            
            new_population[0] = best_individual.copy()
            
            population = np.array(new_population)
            
            # Imprime progresso
            if(generation + 1) % 10 == 0:
                print(f"Geração {generation + 1}/{n_generations}, Melhor ganho médio: {best_fitness:.2f} dB")
        
        return best_individual, best_fitness