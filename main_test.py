
import trainer as tr 
import  population
import genome
import tensorflow as tf

import crossover


N_genomes = 10

Pop =  population.Population(N_genomes) # debe ser la poblacion
graph_glob =  tf.Graph()
input_units = 16
output_units = 4

for i in range(0,N_genomes):
    genome_new = genome.Genome( input_units , output_units , graph=graph_glob )
    Pop.add_genome(genome_new)



train = tr.Trainer(Pop, graph_g = graph_glob )

train.train()

