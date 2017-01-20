
import networkx as nx

import numpy as np
from itertools import izip

import random as rnd

from genome import Genome

class Crossover():

    def __init__(self):

        # crossover
        
        print("Crossover  object")

    def combine_genomes(self , genome1 , genome2):

       # print("Empezando prro")
        matching_edges = []
        disjoint_edges_fitter = []
        excess_edges_fitter = []
       
        g_fitter = None
        g_other = None
        if genome1.calculated_fitness >= genome2.calculated_fitness:
            g_fitter = genome1
            g_other = genome2
        else:
            g_fitter = genome2
            g_other = genome1

        rr = rnd.random()
        r2 = rnd.random()
        for  u , vs in g_fitter.get_edge_iter():
            matching_count = 0
            
            for v , eatt1  in vs.items():
                data1 = eatt1['inN']
                ## find in  the
                
                for u2 , vs2  in g_other.get_edge_iter():
                    for v2 , eatt2  in vs2.items():
                        data2 = eatt2['inN']
                        if data2 == data1:
                            matching_count = matching_count + 1 
                           # print("matching!!")
                            ## tiene el mismo numero de inovacion, elegimos aleatoriamente un edge
                            #rr = rnd.random()
                        
                            atts_inh = None
                            if r2 < 0.5:
                                atts_inh = eatt1
                            else:
                                atts_inh = eatt2
                                
                            #print(atts_inh)
                            matching_edges.append( ( u ,  v , atts_inh )  )
                          
                        #else:
                             #print("no matching prro")
                        
                if matching_count == 0 :
                    # no hubo conteos positivos
                    if g_fitter.inovationNumber <=  g_other.inovationNumber:
                        # disjoint
                        disjoint_edges_fitter.append( ( u ,v , eatt1 )   )
                    else:
                        # exess
                        excess_edges_fitter.append( (u,v, eatt1)  )
                        
                        ## no coinciden

        """                        
        print(matching_edges)
        print(len(matching_edges ))
        print(len(  excess_edges_fitter  ) )
        print(len(  disjoint_edges_fitter  ) )
        """
        Gnew = Genome()
        
        Gnew.clear()
        Gnew.add_nodes_from( g_fitter.get_input_genes() )
        Gnew.add_nodes_from( g_fitter.get_output_genes() )
        Gnew.add_edges_from(matching_edges)
        Gnew.add_edges_from( disjoint_edges_fitter )
        Gnew.add_edges_from( excess_edges_fitter )
        Gnew.set_inovation_number( g_fitter.get_inovation_number() )

        Gnew.set_specie( g_fitter.specie )
        return Gnew
