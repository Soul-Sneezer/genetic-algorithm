import math
import numpy as np

output_file = open("Evolutie.txt", "w")

class Function:
    def __init__(self, a, b, c):
        self.a = a 
        self.b = b 
        self.c = c

    def evaluate(self, x):
        return self.a * (x ** 2) + self.b * x + self.c

class Chromosome:
    def __init__(self, bits, value, num_bits):
        self.bits = bits
        self.value = value
        self.num_bits = num_bits

    def encode_value(self, value, num_bits):
        bit_string = ""

        while value > 0:
            if value % 2 == 0:
                bit_string += "0"
            else:
                bit_string += "1"

            value = int(value / 2)

        while len(bit_string) < num_bits:
            bit_string += "0"

        return reversed(bit_string)

    def decode_bits(self, bits):
        value = 0 
        
        for bit in bits:
            value = value * 2 + (bit == '1')

        return value

class Sim:
    @staticmethod
    def parse_json(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)

                population = data.get("population")
                domain = data.get("domain")
                coefficients = data.get("coefficients")
                precision = data.get("precision")
                crossover_chance = data.get("crossover_chance")
                mutation_chance = data.get("mutation_chance")
                iterations = data.get("iterations")

                parsed_data = {
                    "population": population,
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

    def __init__(self, population, domain, coefficients, precision, crossover_chance, mutation_chance, iterations):
        self.population_size = len(population)
        self.population = population 
        self.domain = domain 
        self.function = Function(coefficients[0], coefficients[1], coefficients[2]) 
        self.precision = precision 
        self.crossover_chance = crossover_chance 
        self.mutation_chance = mutation_chance 
        self.iterations = iterations

        self.num_bits = math.ceil(math.log2((domain[1] - domain[0]) * (10 ** precision)))
        self.d = (domain[1] - domain[0]) / (2 ** num_bits)

    def __init__(self, config_file):
        self.population_size = len(config_file.population)
        self.population = config_file.population 
        self.domain = config_file.domain 
        self.function = Function(config_file.coefficients[0], config_file.coefficients[1], config_file.coefficients[2])
        self.precision = config_file.precision 
        self.crossover_chance = config_file.crossover_chance 
        self.mutation_chance = config_file.mutation_chance 
        self.iterations = config_file.iterations

        self.num_bits = math.ceil(math.log2((self.domain[1] - self.domain[0]) * (10 ** self.precision)))
        self.d = (self.domain[1] - self.domain[0]) / (2 ** num_bits)

    def initialise(self, population_size):
        self.population = np.random.randint(low=0, high=(1 << l), size=population_size)

    def binary_search(low, high, intervals, value):
        mid = (low + high) / 2 
        if value >= intervals[mid][0] and value <= intervals[mid][1]:
            return mid + 1 
        elif intervals[mid] < value:
            return binary_search(mid + 1, high, arr, value)
        else:
            return binary_search(low, mid - 1, arr, value)

    def selection(self): 
        new_generation = []
        intervals = [0.0]
        total_fitness = 0.0 
        total_probability = 0.0
        best_fitness = 0.0
        best_chromosome = population[0] 
        probabilities = np.array(len(population))

        for chromosome in population:
                fitness = function.evaluate(chromosome.value)
                total_fitness += fitness 

                if fitness > best_fitness:
                    best_fitness = fitness 
                    best_chromosome = chromosome

        new_generation.append(best_chromosome)
        
        for chromosome in population:
                probabilities[i] = function.evaluate(chromosome.value) / total_fitness
                total_probability += probabilities[i]
                intervals.append(total_probability)

        while len(new_generation) < population_size: # keep on selecting parents
            parents = []
            
            rand_val = np.rand(0, 1)
            
            i = binary_search(0, 1, probabilities, rand_val) # find which chromosome was selected
            print(i)

            if len(parents) == 2:
                children = crossover(self, parent1, parent2)
                for child in children:
                    new_generation.append(child)

    def crossover(self, parent1, parent2):
        if rand() < crossover_chance:
            pass 
        else: 
            pass

    def mutate(self):
        if rand() < mutation_chance:
            pass 
        else:
            pass

    def execute_iteration(self):
        pass 

    def execute_iterations(self, file_name, show):
        while iterations > 0:
            execute_iteration() 
            iteration = iteration - 1


config = Sim.parse_json("config.json")
if config:
    sim = Sim(config) 
    sim.execute_iterations("Evolutie.txt", show=True)
else:
    print("Failed to load config.")


