import math
import numpy as np
import json
import random
from functools import partial

output_file = open("Evolutie.txt", "w")

def fitness_function(x, a, b, c):
    return a * x ** 2 + b * x + c

polynomial = partial(fitness_function, a=-1, b=1, c=2)

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
    def __init__(self, population_size, domain, num_bits, bit_value):
        self.size = population_size
        self.chromosomes = np.array([Chromosome(random_binary_string(num_bits), num_bits, bit_value, domain) for _ in range(population_size)], dtype=object)

    def __str__(self):
        return f"max: {self.max_fitness} mean: {self.mean_fitness} median: {self.median_fitness}" 

    def __repr__(self):
        return f"max: {self.max_fitness} mean: {self.mean_fitness} median: {self.median_fitness}"

    def initialise(self):
        for chromosome in chromosomes:
            chromosome = np.random.randint(low=0, high=(1 << num_bits), size=population_size)

    def get_max_fitness(self):
        return max(self.chromosomes, key=lambda c: c.fitness) 

    def get_mean_fitness(self):
        return sum(c.fitness for c in self.chromosomes) / self.size

    def get_median_fitness(self):
        sorted_chroms = sorted(self.chromosomes, key=lambda c: c.fitness)
        mid = self.size // 2
        if self.size % 2 == 0:
            return (sorted_chroms[mid - 1].fitness + sorted_chroms[mid].fitness) / 2
        else:
            return sorted_chroms[mid].fitness

    def visualize():
        pass

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
        self.population = Population(population_size, domain, self.num_bits, self.bit_value)
        self.domain = domain 
        self.precision = precision 
        self.crossover_chance = crossover_chance 
        self.mutation_chance = mutation_chance 
        self.iterations = iterations
        
    def binary_search(self, low, high, intervals, value):
        while low < high:
            mid = (low + high) // 2 
            if intervals[mid] <= value < intervals[mid + 1]:
                return mid 
            elif value < intervals[mid]:
                high = mid 
            else: 
                low = mid + 1
        return None

    def selection(self, intervals): 
        rand_val = np.random.rand()  
        i = self.binary_search(0, len(intervals), intervals, rand_val) # find which chromosome was selected
        return self.population.chromosomes[i]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_chance:
            return (parent1, parent2)
        else: 
            crossover_point = np.random.randint(self.num_bits)
            str_list1 = list(map(str, parent1.bits))
            str_list2 = list(map(str, parent2.bits))

            substr = str_list1[crossover_point:]
            str_list1[crossover_point:] = str_list2[crossover_point:]
            str_list2[crossover_point:] = substr
            
            new_parent1 = Chromosome("".join(str_list1), self.num_bits, self.bit_value, self.domain)
            new_parent2 = Chromosome("".join(str_list2), self.num_bits, self.bit_value, self.domain)
            return (new_parent1, new_parent2)

    def mutate(self, chromosome):
        chromosome_list = list(chromosome.bits)
        for bit in chromosome_list:
            if np.random.rand() > self.mutation_chance:
                if bit == '1':
                    bit = '0'
                else:
                    bit = '1'

        return Chromosome("".join(chromosome_list), self.num_bits, self.bit_value, self.domain)

    def execute_iteration(self):
        new_generation = []
        intervals = [0.0]
        total_fitness = 0.0 
        total_probability = 0.0
        best_fitness = 0.0
        best_chromosome = self.population.chromosomes[0] 
        probabilities = np.zeros(self.population_size)

        for chromosome in self.population.chromosomes:
            fitness = chromosome.evaluate(polynomial)
            total_fitness += fitness 

            if fitness > best_fitness:
                best_fitness = fitness 
                best_chromosome = chromosome

        new_generation.append(best_chromosome)
      
        print(f"total_fitness: {total_fitness}")

        i = 0
        for chromosome in self.population.chromosomes:
                probabilities[i] = chromosome.fitness / total_fitness
                print(chromosome.fitness)
                print(probabilities[i])
                total_probability += probabilities[i]
                intervals.append(total_probability)
                i = i + 1

        while len(new_generation) < self.population_size: # keep on selecting parents
            parent1 = self.selection(intervals)
            parent2 = self.selection(intervals)
            print(parent1, parent2)
            children = self.crossover(parent1, parent2)
            mutated_children = [self.mutate(child) for child in children]

            new_generation.extend(mutated_children[:self.population.size - len(new_generation)])
        self.population.chromosomes = new_generation

    def execute_iterations(self, file_name, show):
        while self.iterations > 0:
            self.execute_iteration()
            self.iterations = self.iterations - 1 

config = Sim.parse_json("config.json")
if config:
    sim = Sim(config["population_size"], config["domain"], config["coefficients"], config["precision"], config["crossover_chance"], config["mutation_chance"], config["iterations"]) 
    sim.execute_iterations("Evolutie.txt", show=True)
else:
    print("Failed to load config.")
