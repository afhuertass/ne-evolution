 
import numpy as np
import networkx as nx
import random
from gene import Gene
from conn import Conn
import matplotlib.pyplot as plt

import tensorflow as tf 
# Genome is a a coleccion on genes y sus relaciones.. es un grafo !


class Genome():
    def __init__(self , InputUnits = 16 , OutputUnits = 4 , graph = None):
        self.inovationNumber = 0
        self.calculated_fitness = 0
        
        self.genome = nx.DiGraph()
        #
        N = 16 # input genes
        self.inputGenes = []
        self.outputGenes = []
        self.graph = graph 
      
      
        for i in range(0,InputUnits):
            gn = Gene( 1.0 ,input_gene = True  , response=1.0, graph= self.graph)
            self.inputGenes.append(gn)
            self.genome.add_node(gn)
            
        for i  in range(0, OutputUnits):
            gn =  Gene(0.0, output_gene = True , graph = self.graph)
            self.outputGenes.append( gn )
            self.genome.add_node(gn)


        
        # random conection
       
        self.add_conn_rnd()
        self.add_conn_rnd()

        self.specie = ""

    def set_specie( self, specie_new):

        self.specie = specie_new

    def re_add(self):
        self.tf_session = tf.Session()
        for gene in self.genome.nodes_iter():
            gene.re_add()

            
        
    def add_gene_rnd(self , gene):
        # agregar gene en una position aleatoria
        ## MUTACION FIGURA 3 B, Paper neat
        self.inovationNumber = self.inovationNumber + 1 
        edge = self.get_rand_edge()
        #print(edge)
        weight_1 = random.random()
        weight_2 = random.random()
        
        self.genome.add_edge( edge[0][0] , gene ,  weight=weight_1 )
        self.genome[edge[0][0]][gene]['inN'] = self.inovationNumber
        
        conn1 = Conn( edge[0][0] , weight_1 )
        gene.add_conn(conn1)

        self.inovationNumber = self.inovationNumber + 1 
        self.genome.add_edge( gene , edge[0][1] , weight = weight_2 )
        
        self.genome[gene][edge[0][1]]['inN'] = self.inovationNumber
        
        conn2 = Conn( gene  , weight_2)
        edge[0][1].add_conn(conn2)
        
        self.genome.remove_edge( edge[0][0] , edge[0][1] )
        # edge[0][0] -> nodo origen
        # edge[0][1] -> nodo llegada
        edge[0][1].rm_conn( edge[0][0] )#pasar nodo de origen
        
        
    def add_conn_rnd(self):
        ## MUTACION  ADD CONNECTION
        
        #self.inovationNumber = self.inovationNumber + 1
        
        notOutput = True
        gene1 = self.get_rand_gene()
        gene2 = self.get_rand_gene()
        
        while gene1[0] == gene2[0] and not ( gene1[0].isInput() and gene2[0].isInput() )  :
            ## work it until yo get different genes
            
            gene2 = self.get_rand_gene()

        ## check if connection already exist
        weight = random.random()
        self.add_genes( gene1[0] , gene2[0] ,  weight )
        
        return True
        
    def add_genes(self , gene1 , gene2 , weight_v  = 1.0):
        ##MUTACION AGREGAR GENE
        self.inovationNumber = self.inovationNumber + 1
        # add a weigthed conexion of 1
        
        self.genome.add_edge( gene1 , gene2 , weight = weight_v )
        # so when we add a edge with a weight,
        conn = Conn( gene1 , weight_v )
        gene2.add_conn(conn)
        
        #FIX IT
        self.genome[gene1][gene2]['inN'] = self.inovationNumber

        

    def get_rand_edge(self ):

        edge = random.sample( self.genome.edges() , 1)
        return edge
    
    def get_rand_gene(self):
        
        gene = random.sample(self.genome.nodes() , 1)
        return gene

    def add_gene(self, gn):
        #gn = Gene(0.0)
        self.genome.add_node(gn)
        
  
    def build_graph(self, session = None  , graph_n = None):
        
        if not graph_n:
            print("PLEASE FEED SOME GRAPH")
            return True
        
        
        for gene in self.outputGenes:
            #gene.activation2( graph = graph_n )
            self.activate_predecessors( gene  , graph_n = graph_n )
            
            #gene.activation2(graph = graph_n )
            

    def activate_predecessors(self , gene , graph_n = None ):
        # change to active predecessors
        if gene.isAdded():
            return 
        ws = []
        xs = []
       # print( len( self.genome.predecessors(gene)  ) )
        for pred in self.genome.predecessors(gene  ):
            # pred es nodo precedesor
            # por cada nodo precedessor debemos colectar la operacion
            # peso*ops nodo
            if not pred.isAdded():
                #ws.append( self.genome[pred][gene]['weight'] )
                
                self.activate_predecessors( pred  , graph_n = graph_n  )
                
                pred.set_added_to_graph(added = True)
                #pred.activation2( graph = graph )
                ws.append( self.genome[pred][gene]['weight'] )
                xs.append( pred.act  )
                

        print(xs)
        print(ws)
        with graph_n.as_default():
            if gene.input_gene:
                print("PLACE HOLDER")
                gene.act = tf.placeholder( tf.float32 , shape = () )
                gene.set_added_to_graph(added = True )

            if len(ws) == 0 or len(xs) == 0:
                gene.act = tf.Variable(tf.zeros( [] ) , dtype=tf.float32)
                gene.set_added_to_graph(added = True )
            else:
                ws = tf.Variable( np.array( ws) ,  dtype = tf.float32 )
                xs = tf.pack( xs , axis = 0 )
            
                gene.act = tf.sigmoid(tf.reduce_sum( tf.mul( xs, ws) )   )
                gene.set_added_to_graph(added = True )

    def update_conns(self, graph_n = None):

        with graph_n.as_default():
            for gene in self.genome.nodes_iter():
                gene.clear_conns()
                
                for pred in self.genome.predecessors(gene):
                    weight = self.genome[pred][gene]['weight']
                    NewConn = Conn( pred , weight )
                    gene.add_conn( NewConn )
                   
                    
    def launch_session(self, get_value = False , get_index = False, inputs = None, session = None):

        if not session:
            return False
        
        print("respuesta del grafo")
        if not inputs:
            return False

        placeholders = self.get_list_placeholders()
        
        out = []
        out_val = []
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            
         #init = tf.initialize_all_variables()
        resp = 0
        
        print( inputs )
        print( placeholders )
            
        #with session  as sess:
        for node in self.outputGenes:
                #sess.run( init , feed_dict =   {} )
            if node.act == None:
                print("this happen")
                continue 
            a =session.run( node.act , feed_dict  = { i : d for i,d in zip( placeholders ,inputs) } )
                #a =  sess.run( node.act , feed_dict  = { i : d for i,d in zip( placeholders ,inputs) }   ) 
                #a =  sess.run( node.act , feed_dict  = { placeholders[0] : 1.0 ,  }   ) 
            out_val.append( a) 
            out.append( node.act )

            

        if len( out ) == 0:
            # ningun nodo de salida dio respuesta
            
            print (" OUT NODEs GIVE NO  ANWSER")
            return False
        if get_index:
            #with session as sess:
            out = tf.pack( out , axis = 0)
            out = tf.argmax(out , axis = 0 )
            resp = int (  session.run( out ) )
            # la respuesta va a ser el indice del de mayor valor
        if get_value:
            resp = out_val
        return resp
            

    def get_list_placeholders(self):

        inputs_placeholders = []
        for gene in self.inputGenes:

            inputs_placeholders.append( gene.act )

        return inputs_placeholders
        
    def init_variables(self):
        init = tf.global_variables_initializer()
        self.tf_session( init )


    def close_session(self):

        self.tf_session.close()
    
    def set_inovation_number(self, inovationN):

        self.inovationNumber = inovationN

    def  get_inovation_number(self):

        return ( self.inovationNumber )
    
    def get_edge_iter(self):

        return self.genome.adjacency_iter()

    def clear(self):

        self.genome.clear()

    def add_edges_from(self, nedges):

        self.genome.add_edges_from(  nedges )

    def add_nodes_from(self , nnodes):

        self.genome.add_nodes_from(nnodes)
        
    def mutate_perturbation(self):

        #do something
        rnd1 = random.random()
        #rnd2 = random.random()
        
        for u , vs in self.get_edge_iter():
            for v , atts in vs.items():
                 
                new_weight = atts['weight'] + ( rnd1 )
                self.genome[u][v]['weight'] = new_weight
        return 0.0

    def mutate_new_rnd(self):

        #change all weights for random

        for u, vs   in self.get_edge_iter():
            ## cambiar el peso del edge
            for v , atts  in vs.items():
                new_weight = random.random()
                self.genome[u][v]['weight'] = new_weight
                # 
        return 0.0
    def set_fitness(self, new_fitness):

        self.calculated_fitness = new_fitness
    
    def get_input_genes(self):

        return self.inputGenes

    def get_output_genes(self):

        return self.outputGenes
    def mutate_new_link(self):
        
        self.add_conn_rnd()
        return 0.0

    def mutate_new_node(self):

        Gnew = Gene( 0.0 , graph = self.graph)
        self.add_gene_rnd( Gnew )
        return 0.0

    def feed(self ,  list_inputs ):
        #WRONGGG 
        index = 0
        
        for inputGene in self.inputGenes:
            
            inputGene.assign_value(  list_inputs[index]  )
            index = index +  1 
    def save_to_pickle(self, path_to):
        #print( type( self.genome) )
        #new = nx.convert_node_labels_to_integers(self.genome )
        mapping = {}
        new_input_node = "input_"
        new_output_node = "output_"
        index_input_nodes = 0 
        index_output_nodes = 0
        index_nodes = 0
        for node in self.genome.nodes_iter():
            if node.isInput():
                inputNodeName = new_input_node + str(index_input_nodes)
                mapping[node] = inputNodeName
                index_input_nodes = index_input_nodes + 1
                continue
            if node.isOutput():
                outputNodeName = new_output_node + str(index_output_nodes)
                mapping[node] = outputNodeName
                index_output_nodes = index_output_nodes + 1
                continue

            mapping[node] = index_nodes
            index_nodes = index_nodes + 1

        new = nx.relabel_nodes(self.genome , mapping )
        nx.write_gpickle( G =new  , path=path_to)
        
    def Draw(self):

        nx.draw(self.genome  )
        plt.show()


    
        

