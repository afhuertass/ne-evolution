
import trainer as tr 
import  population
import genome
import tensorflow as tf 
N_genomes =10


Pop =  population.Population(N_genomes) # debe ser la poblacion

#gen = genome.Genome()

#gen.save_to_pickle("papu.pickle")

graph_glob =  tf.Graph()
input_units = 2
output_units = 1 
for i in range(0,N_genomes):
    genome_new = genome.Genome( input_units , output_units , graph=graph_glob )
    Pop.add_genome(genome_new)

#genome = genome.Genome()

"""
Pop.organize_species()
Pop.reproduce()


Pop.organize_species()

Pop.reproduce()

Pop.organize_species()

Pop.reproduce()

Pop.organize_species()

Pop.reproduce()
"""
train = tr.Trainer(Pop, graph_g = graph_glob )

train.train()
 



