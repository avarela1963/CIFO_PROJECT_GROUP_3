import random as rndm
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from sudoku_data import y

"""
Part 1: Defining Genes and Chromosomes
"""

def execute_line(initial=None):
    if initial is None:
        initial = [0] * 9
    mapp = {}
    gene = list(range(1, 10))
    rndm.shuffle(gene)
    for i in range(9):
        mapp[gene[i]] = i
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            temp = gene[i], gene[mapp[initial[i]]]
            gene[mapp[initial[i]]], gene[i] = temp
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]
    return gene

def execute_individual(initial=None):
    if initial is None:
        initial = [[0 for _ in range(9)] for _ in range(9)]
    chromosome = []
    for i in range(9):
        chromosome.append(execute_line(initial[i]))
    return chromosome

"""
Part 2: Making First Generation
"""

def make_population(count, initial=None):
    if initial is None:
        initial = [[0 for _ in range(9)] for _ in range(9)]
    population = []
    for _ in range(count):
        population.append(execute_individual(initial))
    return population

"""
Part 3: Fitness Function
The fitness function calculates how "fit" a chromosome (puzzle) is based on:

For each column: Subtract (number of times a number is seen) - 1 from the fitness for that number
For each 3x3 square: Subtract (number of times a number is seen) - 1 from the fitness for that number 
The higher the fitness, the closer the puzzle is to being solved.
"""

def get_fitness(chromosome):
    """Calculate the fitness of a chromosome (puzzle)."""
    fitness = 0
    for j in range(9):  # For each row
        seen = {}
        for i in range(9):  # Check each cell in the column
            if chromosome[j][i] in seen:
                seen[chromosome[j][i]] += 1
            else:
                seen[chromosome[j][i]] = 1
        for key in seen:  # Subtract fitness for repeated numbers
            fitness -= (seen[key] - 1)
    for i in range(9): # For each column
        seen = {}
        for j in range(9): # Check each cell in the column
            if chromosome[j][i] in seen:
                seen[chromosome[j][i]] += 1
            else:
                seen[chromosome[j][i]] = 1
        for key in seen: # Subtract fitness for repeated numbers
            fitness -= (seen[key] - 1)
    for m in range(3): # For each 3x3 square
        for n in range(3):
            seen = {}
            for i in range(3 * n, 3 * (n + 1)):  # Check cells in 3x3 square
                for j in range(3 * m, 3 * (m + 1)):
                    if chromosome[j][i] in seen:
                        seen[chromosome[j][i]] += 1
                    else:
                        seen[chromosome[j][i]] = 1
            for key in seen: # Subtract fitness for repeated numbers
                fitness -= (seen[key] - 1)
    return fitness

def crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = rndm.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        else:
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2

def mutation(ch, pm, initial):
    for i in range(9):
        x = rndm.randint(0, 100)
        if x < pm * 100:
            ch[i] = execute_line(initial[i])
    return ch

def r_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rndm.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool

# Constants
POPULATION = 10000
REPETITION = 50
PM = 0.01
PC = 0.95

# Main genetic algorithm function
def genetic_algorithm(initial_puzzle):
    max_fitness_values = []  # To store maximum fitness values
    population = make_population(POPULATION, initial_puzzle)

    try:
        for i in range(REPETITION):
            print(f"Iteration: {i}")

            # Calculate the fitness of the current population
            fitness_and_individuals = [(get_fitness(c), c) for c in population]
            best_fitness_before, best_individual_before = max(fitness_and_individuals, key=lambda x: x[0])
            print(f"Best fitness before: {best_fitness_before}")

            # Save the current population in case we need to revert
            previous_population = deepcopy(population)

            # Selection of the mating pool
            mating_pool = r_get_mating_pool(population)
            rndm.shuffle(mating_pool)
            population = get_offsprings(mating_pool, initial_puzzle, PM, PC)

            # Calculate the fitness of the new population
            fitness_and_individuals_after = [(get_fitness(c), c) for c in population]
            best_fitness_after, best_individual_after = max(fitness_and_individuals_after, key=lambda x: x[0])
            print(f"Best fitness after: {best_fitness_after}")

            # If the best fitness in the new population is worse, revert to the previous population
            if best_fitness_after < best_fitness_before:

                population = previous_population
                best_fitness_after = best_fitness_before
                best_individual_after = best_individual_before

            # Update the max fitness values
            max_fitness_values.append(best_fitness_after)
            print(f"Max fitness current: {best_fitness_after}")
            print()

            # Check for termination condition
            if best_fitness_after == 0:
                return population, max_fitness_values

    except KeyboardInterrupt:
        print("Execution interrupted by user.")

    return population, max_fitness_values  # Added return statement here

# Initialize an empty DataFrame
df = pd.DataFrame()

# Run the algorithm multiple times and store the results in the DataFrame
num_runs = int(input("Enter the number of runs: "))
success_count = 0
for run in range(num_runs):
    tic = time.time()
    try:
        r, max_fitness_values = genetic_algorithm(y)
        if max_fitness_values[-1] == 0:
            success_count += 1
    except KeyboardInterrupt:
        print(f"Run {run + 1} interrupted.")
        break
    toc = time.time()
    print(f"Time taken for run {run + 1}: {toc - tic}")

    # Add the results of this run to the DataFrame using pd.concat
    df = pd.concat([df, pd.Series(max_fitness_values).to_frame().T], ignore_index=True)

# Calculate success rate
success_rate = success_count / num_runs
print(f"Success rate: {success_rate * 100:.2f}%")

# Replace NaN values with zero
df.fillna(0, inplace=True)

# Calculate the average fitness for each iteration
df.loc['Average'] = df.mean()

# Save the DataFrame to a CSV file
df.to_csv(f'fitness_results_{POPULATION}.csv', index=False)

# Print the DataFrame and all its columns
pd.set_option('display.max_columns', None)  # Ensure all columns are printed
print(df)


