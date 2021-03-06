

import numpy as np
import time
import scraper as scraper
import tensorflow as tf 

from pympler import tracker 
import resource


class Trainer:

    """
    Esta clase sera el corazon del programa recibe la poblacion de genes a ser evaluada. se requiere realizar lo siguiente con el fin de hacerla funcionar

    1.  Crear clase capturadora de imagenes.esta  necesita un punto y ancho y alto y captura la pantalla de esas coordenadas. 

    2. crear clase de proceso de imagen, esta clase ya existe en image_processing.py , toma la imagen del capturador y regresa la lista

    3. en el metodo train se entrena cada genoma. se repite para cada genoma y tambien en N pasos ( definido por el numero de jugadas maximas) 
    """
    def __init__(self, population , graph_g = None):

        self.population = population
        self.max_moves = 10
        self.max_gen = 3

        self.driver_data = scraper.DriverData()
        #self.driver_data = scraper.XORDriver()
        
        #self.driver_data.get_train_list()
        self.graph = graph_g 
        self.tf_sess = tf.Session( graph = graph_g )

        self.memory_tracker = tracker.SummaryTracker()
        self.memory_tracker.print_diff()
        
    def train(self):

        #for genome in self.population.get_genes():
          #  genome.update_conns()
           # genome.build_graph(session = self.tf_sess)

        #tf.get_default_graph().finalize()


        results = open( "./xor_test1.dat" ,  "w")

        broken = False 
        for gen in range(0, self.max_gen ):
            max_score_gen = 0
            genome_count = 1

            total_score = 0
            N_genomes = self.population.N
            new_graph = tf.Graph()
            new_session = tf.Session( graph = new_graph)
            for genome in self.population.get_genes():
                #genome.Draw()
                #genome.update_conns( graph_n = new_graph )
                genome.build_graph( graph_n = new_graph)
                
                
            new_session = tf.Session( graph = new_graph)
            print("Generacion: "  + str(gen) )
            print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for genome in self.population.get_genes():
                #print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                #genome.update_conns()
                #genome.build_graph( graph_n = new_graph )
                outputs = []
                #print("########################")
                #print("Generacion: "  + str(gen) )
                self.driver_data.new_game()
                for step in range(0, self.max_moves ):
                    #print("Generacion: "  + str(gen) )
                    print("Individuo No:" + str(genome_count) )
                    #print("jugadas:" + str(step) )
                    #print("waitting response... ")
                    list_train = []
                    try:
                        list_train = self.driver_data.get_train_list( step )
                    except:
                        broken = True 
                        break

                    #print (list_train)
                        #print("response given  ")
                    response_genome = genome.launch_session(get_index = True , inputs= list_train, session= new_session )
                    if not response_genome:
                        # no hubo respuesta de ningun nodo
                        response_genome = [ -1 ]
                    #print(response_genome)
                    outputs.append( response_genome[0]  )
                    self.handle_response( response_genome )
                    # las teclas fueron presionadas
                    # esperar un momento ( que se acomode el score)

                time.sleep(0.5)
               
                score = self.driver_data.read_score( outputs )
               
                
                #print("FITNESS  individio:" + str( score ))
                genome.set_fitness( score )
                #print("#################")
                
                #genome.close_session()
                genome_count = genome_count +1 
                self.driver_data.new_game()
                
                if score  > max_score_gen:
                    max_score_gen = score

                total_score = total_score + score 
                    
            # o asignar el fit de acuerdo a algo
            #tf.get_default_graph().finalize()
            #self.population.organize_species()
            #with self.graph.as_default():
            #tf.reset_default_graph()
            avg_score = total_score/N_genomes
            to_string = str(gen) + " "  + str(avg_score) + " " + str(max_score_gen)+ "\n"
            results.write( to_string )

            for genome in self.population.get_genes():
                print("Name:" + str(genome.name))
                print("Fitness:" + str(genome.calculated_fitness))
            
            
            self.population.organize_species()
            #self.population.mutate()
            self.population.reproduce()
            
            self.population.mutate()
            
            #self.memory_tracker.print_diff()
          
            
            self.population.re_add()                  
           # self.population.update_conns()

           
           
            #self.population.re_add()
            #self.population.re_add()
            print("Max_score:" + str(max_score_gen ))


            if broken:
                results.write("problem with memory  ")
                break
            
        # fin de esa generacion

        
        results.close()
        
        ## fin del entrenamiento en esa generacion
        self.population.save_champions()
        """
        for genome in self.population.get_genes():

            genome.close_session()
        """

    def handle_response(self,int_key ):

        # do something with the response, press the keys or whatever
        self.driver_data.handle_response( int_key )
        
        #rr = self.driver_data.send_keystroke( int_key )
        
        return True
