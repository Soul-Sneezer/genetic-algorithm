import math
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from functools import partial

output_file = open("Evolutie.txt", "w")

def fitness_function(x, a, b, c):
    return a * x ** 2 + b * x + c

class Chromosome:
    def __init__(self, bits, num_bits, bit_value, domain):
        self.bits = bits
        self.num_bits = num_bits 
        self.bit_value = bit_value
        self.domain = domain
        self.value = self.decode_bits(bits)
        self.fitness = None

    def  __str__(self): 
        return f"bits: {self.bits} value: {self.value} fitness: {self.fitness}"

    def __repr__(self):
        return f"bits: {self.bits} value: {self.value} fitness: {self.fitness}"

    def evaluate(self, func):
        self.fitness = func(self.value)
        return self.fitness

    def encode_value(self, value, num_bits):
        bit_string = ""

        while value > 0:
            if value % 2 == 0:
                bit_string += '0'
            else:
                bit_string += '1'

            value = int(value / 2)

        while len(bit_string) < num_bits:
            bit_string += '0'

        return reversed(bit_string)

    def decode_bits(self, bits):
        value = 0.0
        
        for bit in bits:
            value = value * 2 + (bit == '1') * self.bit_value

        return self.domain[0] + value

def random_binary_string(n):
    return ''.join(random.choice('01') for _ in range(n))

class Population:
    def __init__(self, population_size, domain, num_bits, bit_value, function):
        self.size = population_size
        self.chromosomes = np.array([Chromosome(random_binary_string(num_bits), num_bits, bit_value, domain) for _ in range(population_size)], dtype=object)
        self.max_fitness = 0.0 
        self.mean_fitness = 0.0 
        self.median_fitness = 0.0
        self.function = function

    def initialise(self):
        for chromosome in chromosomes:
            chromosome = np.random.randint(low=0, high=(1 << num_bits), size=population_size)

    def get_max_fitness(self):
        self.max_fitness = max(c.evaluate(self.function) for c in self.chromosomes) 
        return self.max_fitness 

    def get_mean_fitness(self):
        self.mean_fitness = sum(c.evaluate(self.function) for c in self.chromosomes) / self.size
        return self.mean_fitness 

    def get_median_fitness(self):
        sorted_chroms = sorted(self.chromosomes, key=lambda c: c.evaluate(self.function))
        mid = self.size // 2
        if self.size % 2 == 0:
            self.median_fitness = (sorted_chroms[mid - 1].fitness + sorted_chroms[mid].fitness) / 2
        else:
            self.median_fitness = sorted_chroms[mid].fitness
        return self.median_fitness
    
    def __str__(self):
        return f"max: {self.get_max_fitness()} mean: {self.get_mean_fitness()} median: {self.get_median_fitness()}" 

    def __repr__(self):
        return f"max: {self.get_max_fitness()} mean: {self.get_mean_fitness()} median: {self.get_median_fitness()}"

class Sim:
    @staticmethod
    def parse_json(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)

                population_size = data.get("population_size")
                domain = data.get("domain")
                coefficients = data.get("coefficients")
                precision = data.get("precision")
                crossover_chance = data.get("crossover_chance")
                mutation_chance = data.get("mutation_chance")
                iterations = data.get("iterations")

                parsed_data = {
                    "population_size": population_size,
                    "domain": domain,
                    "coefficients": coefficients,
                    "precision": precision,
                    "crossover_chance": crossover_chance,
                    "mutation_chance": mutation_chance, 
                    "iterations": iterations
                }

                return parsed_data
        except Exception as e:
            print(f"{e} when opening {file_path}")
            return None

    def __init__(self, population_size, domain, coefficients, precision, crossover_chance, mutation_chance, iterations):
        self.num_bits = math.ceil(math.log2((domain[1] - domain[0]) * (10 ** precision)))
        self.bit_value = (domain[1] - domain[0]) / (2 ** self.num_bits)
   
        self.population_size = population_size
        self.domain = domain 
        self.precision = precision 
        self.crossover_chance = crossover_chance 
        self.mutation_chance = mutation_chance 
        self.iterations = iterations
        self.polynomial = partial(fitness_function, a=coefficients['a'], b=coefficients['b'], c=coefficients['c'])
        self.population = Population(population_size, domain, self.num_bits, self.bit_value, self.polynomial)

        self.generations = []
        
    def binary_search(self, low, high, intervals, value, iteration):
        while low < high:
            mid = (low + high) // 2 
            if intervals[mid] <= value < intervals[mid + 1]:
                return mid 
            elif value < intervals[mid]:
                high = mid 
            else: 
                low = mid + 1
        return None

    def selection(self, intervals, iteration): 
        rand_val = np.random.rand()  
        i = self.binary_search(0, len(intervals), intervals, rand_val, iteration) # find which chromosome was selected
        
        if iteration == 0: 
            print(f"random generated number:{rand_val} interval: {i}")

        return self.population.chromosomes[i]

    def crossover(self, parent1, parent2, iteration):
        if np.random.rand() > self.crossover_chance:
            return (parent1, parent2)
        crossover_point = np.random.randint(1, self.num_bits - 1)
        
        bits1 = parent1.bits[:crossover_point] + parent2.bits[crossover_point:]
        bits2 = parent2.bits[:crossover_point] + parent1.bits[crossover_point:]

        if iteration == 0: 
            print(f"parents: {parent1}, {parent2}")
            print(f"crossover point: {crossover_point}")
            print(f"children: {bits1}, {bits2}")

        return (
            Chromosome(bits1, self.num_bits, self.bit_value, self.domain),
            Chromosome(bits2, self.num_bits, self.bit_value, self.domain)
        )

    def mutate(self, chromosome, iteration):
        chromosome_list = list(chromosome.bits)
        for bit in chromosome_list:
            if np.random.rand() > self.mutation_chance:
                if bit == '1':
                    bit = '0'
                else:
                    bit = '1'
        chromosome.bits = "".join(chromosome_list)
        #return Chromosome("".join(chromosome_list), self.num_bits, self.bit_value, self.domain)

    def execute_iteration(self, iteration):
        if iteration == 0:
            for c in self.population.chromosomes:
                c.evaluate(self.polynomial)
                print(c)

        new_generation = []
        intervals = [0.0]
        total_fitness = 0.0 
        total_probability = 0.0
        best_fitness = 0.0
        best_chromosome = self.population.chromosomes[0] 
        probabilities = np.zeros(self.population_size)

        for chromosome in self.population.chromosomes:
            fitness = chromosome.evaluate(self.polynomial)
            total_fitness += fitness 

            if fitness > best_fitness:
                best_fitness = fitness 
                best_chromosome = chromosome

        new_generation.append(best_chromosome)
      
       # print(f"total_fitness: {total_fitness}")

        i = 0
        for chromosome in self.population.chromosomes:
                probabilities[i] = chromosome.fitness / total_fitness
                if iteration == 0:
                #print(chromosome.fitness)
                    print(f"probability: {probabilities[i]}")
                total_probability += probabilities[i]
                intervals.append(total_probability)
                i = i + 1

        if iteration == 0:
            print("intervals:")
            for interval in intervals: 
                print(interval) 

        while len(new_generation) < self.population_size: # keep on selecting parents
            parent1 = self.selection(intervals, iteration)
            parent2 = self.selection(intervals, iteration)
            #print(parent1, parent2)
            children = self.crossover(parent1, parent2, iteration)
            
            new_generation.extend(children[:self.population.size - len(new_generation)])
      
        if iteration == 0:
            print(f"Population after crossovers: {new_generation}")
        
        for child in new_generation:
            self.mutate(child, iteration)
        if iteration == 0:
            print(f"Population after mutations: {new_generation}")

        self.population.chromosomes = new_generation
        self.generations.append([self.population.get_max_fitness(), self.population.get_mean_fitness(), self.population.get_median_fitness()])
        print(self.population)

    def execute_iterations(self, file_name, show):
        iteration = 0
        while iteration < self.iterations:
            self.execute_iteration(iteration)
            iteration = iteration + 1

    def plot_results(self):
        max_fitness_list = [] 
        mean_fitness_list = []
        median_fitness_list = []
        for pop_state in self.generations:
            max_fitness_list.append(pop_state[0])
            mean_fitness_list.append(pop_state[1])
            median_fitness_list.append(pop_state[2])

        generation_numbers = list(range(1, self.iterations + 1))

        plt.figure(figsize=(10, 6))

        plt.plot(generation_numbers, max_fitness_list, label="Max Fitness", color="red", linewidth=2)

        plt.plot(generation_numbers, mean_fitness_list, label="Mean Fitness", color="blue", linestyle="--")

        plt.plot(generation_numbers, median_fitness_list, label="Median Fitness", color="green", linestyle=":")

        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Genetic Algorithm Fitness Over Generations")

        plt.legend()
        plt.grid(True) 
        plt.show()

config = Sim.parse_json("config1.json")
if config:
    sim = Sim(config["population_size"], config["domain"], config["coefficients"], config["precision"], config["crossover_chance"], config["mutation_chance"], config["iterations"]) 
    sim.execute_iterations("Evolutie.txt", show=True)
    sim.plot_results()
else:
    print("Failed to load config.")
