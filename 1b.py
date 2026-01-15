import streamlit as st
import numpy as np

st.title("Genetic Algorithm Bit Pattern Generator")

POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 40

def fitness(chromosome):
    return CHROMOSOME_LENGTH - abs(np.sum(chromosome) - TARGET_ONES)

population = np.random.randint(2, size=(POPULATION_SIZE, CHROMOSOME_LENGTH))

best_fitness_history = []

for gen in range(GENERATIONS):
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_fitness_history.append(np.max(fitness_scores))

    selected = population[np.argsort(fitness_scores)[-POPULATION_SIZE//2:]]

    offspring = []
    for _ in range(POPULATION_SIZE // 2):
        p1, p2 = selected[np.random.randint(len(selected), size=2)]
        cp = np.random.randint(1, CHROMOSOME_LENGTH-1)
        child = np.concatenate([p1[:cp], p2[cp:]])
        offspring.append(child)

    offspring = np.array(offspring)

    mutation_mask = np.random.rand(*offspring.shape) < 0.01
    offspring = np.logical_xor(offspring, mutation_mask).astype(int)

    population = np.vstack((selected, offspring))

best_idx = np.argmax([fitness(ind) for ind in population])
best_solution = population[best_idx]

st.subheader("Best Chromosome")
st.write("".join(map(str, best_solution)))
st.write("Number of Ones:", np.sum(best_solution))
st.write("Fitness Score:", fitness(best_solution))

st.subheader("Fitness Over Generations")
st.line_chart(best_fitness_history)
