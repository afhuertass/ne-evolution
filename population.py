
from __future__ import division

import numpy as np
import tensorflow as  tf
import random
import crossover

import string

"""
Pasos al crear el objeto: 


"""

class Population():

    def __init__(self, N ):# n genomes, population size

        # inicializar N genomas...
        # aqui obviamente hacen falta cosas, como las propieades de
        """
        de cada genoma 

        """
        #
        # lista vacia de los genomas, 
        self.population = [] # lista de todos los genomas , inicialmente se puebla esta lista
        
        #species es un placeholder para clasificar los genomas, basicamente toca determinar cada genome a que especie pertenece... 
        self.species = []

        self.N = N  # numero de individuos
        ## parametros
        
        self.prob_mutation = 0.8 # probabilidad de mutacion

        self.prob_mutation_pert = 0.9 # probabildiad de mutar perturbando los valores

        self.prob_new_link = 0.3
        self.prob_new_node = 0.5

        self.c1 = 0.1
        self.c2 = 0.1
        self.c3 = 0.3

        self.compatibility_thr = 1.0
        

        # crossover system
        self.cross = crossover.Crossover()
        
    def mutate(self):

        ## hacer mutar a la pipol
        for genome in self.population:

            nrd = random.random()
            if nrd <= self.prob_mutation :
                # debe mutar
                print("MUTAAAAAAAAAAAAAAAA")
                nnrd = random.random()
                if nnrd <= self.prob_mutation_pert:
                    genome.mutate_perturbation()
                else:
                    genome.mutate_new_rnd()

            nrd = random.random()

            if nrd <= self.prob_new_link :

                genome.mutate_new_link()
                print("MUTAAAAAAAAAAAAAAAA")
            nrd = random.random()
            if nrd <= self.prob_new_node:

                genome.mutate_new_node()

    def update_conns(self):

        for genome in self.population:
            genome.update_conns()
            
    def re_add(self):

        for genome in self.population:

            genome.re_add()

    def organize_species(self):

        for genome in self.population:
            assigned = False
            for specie in self.species: # specie es una lista de strings

                # obtenemos un gen representativo
                grepre = next((x for x in self.population if x.specie == specie ), None)
                if not grepre:
                    continue
                
                d = self.compatibility(genome , grepre)
                
                if d <= self.compatibility_thr:
                    #agregar a esa especie
                    genome.set_specie( specie ) # el gen correponde a dicha especie
                    
                    assigned = True
            if not assigned: # no hubo especie a la cual meter al joven
                # crear nueva lista
                new_specie = self.new_specie( )
                genome.set_specie( new_specie )
                self.species.append( new_specie )
                #agregas especie a la lista de especies
                #self.species.append( new_specie )

        print("numero especies")
        print(len(self.species) )
        
        
    def reproduce(self):
        #ordenamos por
        new_species = []
        new_population = []

        all_species = []
        
        for specie in self.species:
            genomes_in_specie = []
            for genome in self.population:

                if specie == genome.specie:
                    genomes_in_specie.append( genome )
            print("FUUUUU")
            if len(genomes_in_specie) > 0:
                all_species.append( genomes_in_specie )
        
        for specie in all_species:
            new_specie = []
            specie.sort( key=lambda x : x.calculated_fitness , reverse=True)
            
            champion = specie[0] 
            #new_specie.append( champion )  el campeon pasa sin
            new_population.append( champion )
            reproducing = True
            specie_size = len(specie)
            n_offspring = 0
            while reproducing:
                # generar offspring
                #parent1 = random.choice(specie)
                parent2 = random.choice(specie)
                offspring = self.cross.combine_genomes(champion,parent2)
                #new_specie.append( offspring )

                new_population.append( offspring  )
                
                n_offspring = n_offspring + 1
                if  n_offspring >= specie_size-1 :
                    reproducing = False
                print("reproducing")
                print(n_offspring)
                print( specie_size )
            new_species.append( new_specie )

        #self.species = []
        #self.species = new_species[:]

        self.population = []
        self.population = new_population [:]
                
    def compatibility(self , genome1 , genome2 ):
        N = 0 # total genes
        disjoint_genes = 0
        excess_genes = 0

        match_genes = 0
        weight_total_diff =  0
        for u , vs in genome1.get_edge_iter():
            for v , eatts1 in vs.items():
                inovation1 = eatts1['inN']
                N = N +1 
                for u2 , vs2 in genome2.get_edge_iter():
                    for v2 , eatts2 in vs2.items():
                        inovation2 = eatts2['inN']
                        
                        if inovation1 == inovation2:
                            w2 = eatts2['weight']
                            w1  = eatts1['weight']
                            weight_total_diff =  weight_total_diff + abs(w2-w1)
                            
                            match_genes = match_genes + 1
                if match_genes == 0:
                    
                    if genome1.inovationNumber <= genome2.inovationNumber:
                        disjoint_genes = disjoint_genes + 1
                    else:
                        excess_genes = excess_genes + 1 


        if N < 20 :
            N=1
        if match_genes == 0:
            distance  =  (self.c1*disjoint_genes)/N + (self.c2*excess_genes)/N 
        else :
            distance = (self.c1*disjoint_genes)/N + (self.c2*excess_genes)/N + weight_total_diff/match_genes
            

        return distance

            
                    
    def add_genome(self, new_genome):

        self.population.append( new_genome )
        
    def get_genes(self):

        return (self.population)

    
    def new_specie(self, size=6, chars=string.ascii_uppercase + string.digits):
        
        return ''.join(random.choice(chars) for _ in range(size)  )

    def save_champions(self):
        all_species = [] 
        for specie in self.species:
            genomes_in_specie = []
            for genome in self.population:

                if specie == genome.specie:
                    genomes_in_specie.append( genome )

            if len(genomes_in_specie) > 0 :
                all_species.append( genomes_in_specie )
            
        champs = []
        for specie in all_species:
            new_specie = []
            specie.sort( key=lambda x: x.calculated_fitness , reverse=True)

            champs.append( specie[0] )

        index = 0
        for champ in champs:
            string_name = "specie_"+str(index)+".pickle"
            
            champ.save_to_pickle( string_name )
            index = index +1 

   

        
