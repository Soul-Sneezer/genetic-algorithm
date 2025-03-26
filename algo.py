import math
import numpy as np

output_file = open("Evolutie.txt", "w")

class Function:
    def __init__(self, a, b, c):
        self.a = a 
        self.b = b 
        self.c = c

    def f(x):
        return self.a * (x ** 2) + self.b * x + self.c

class Chromosome:
    def __init__(self, bits, value):
        self.bits = bits
        self.value = value

class Sim:
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

        self.l = math.ceil(math.log2((b - a) * (10 ** p)))
        self.d = (domain.end - domain.start) / (2 ** l)

    def initialise(population_size):
        self.population = np.random.randint(low=0, high=(1 << l), size=population_size)

    def selection():
        probabilities = np.array(len(population))
        total_probability = 0
        for chromosome in population:
            probabilities[i] = function.f(chromosome.value)
            total_probability += probabilities[i]



    def crossover():
        if rand() < crossover_chance:
            pass 
        else: 
            pass

    def mutate():
        if rand() < mutation_chance:
            pass 
        else:
            pass


