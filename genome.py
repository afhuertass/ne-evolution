 
import numpy as np
import networkx as nx
import random
from gene import Gene
from conn import Conn
import matplotlib.pyplot as plt
import string

import tensorflow as tf 
# Genome is a a coleccion on genes y sus relaciones.. es un grafo !


class Genome():
    def __init__(self , InputUnits = 16 , OutputUnits = 4 , graph = None , N_hidden = 1 ):
        self.inovationNumber = 0
        self.calculated_fitness = 0
        
        self.genome = nx.DiGraph()
        #
        """
        Los genomas deben contener un nuevo parametro, numero de  unidades ocultas
        por defecto es 1,  una nueva variable numerica indicando la capa a la que pertenece la unidad sera agregada

        

        las unidades de input tienen por valor N = 0 
        las unidades output tienen valor de N +1 

        el proceso de agregar una conexion: toma un nodo en la capa k 
        y lo conecta a un nodo en la capa k + 1. k debe ser un aleatorio entre cero y  N 
         paso 1 : obtener un aleatorio entre 0 y N, llamarlo k  
         obtener un nodo aleatorio de la capa k. 
         obtener un nodo aleatorio en la capa k +1 . si no hay una conexion agregarla. si la hay volver al paso anterior
         conectar los dos.

        de esta forma las redes creadas son siempre feedforward and there is no need for special modification of the other process lets start with it.



        el proceso de agregar un nodo entre coneccion. no requiere modificacion.

       
        """
        N = 16 # input genes
        self.inputGenes = []
        self.outputGenes = []
        self.graph = graph 

        self.HLayers = N_hidden

        
        #adding  inputnodes
        for i in range(0,InputUnits):
            gn = Gene( 1.0 ,input_gene = True  , response=1.0, graph= self.graph, indexLayer = 0 )
            self.inputGenes.append(gn)
            self.genome.add_node(gn)

        #adding outputnodes
        for i  in range(0, OutputUnits):
            gn =  Gene(0.0, output_gene = True , graph = self.graph, indexLayer = N_hidden + 1 )
            self.outputGenes.append( gn )
            self.genome.add_node(gn)



        bias = Gene(0.0 , bias_gene=True , graph = self.graph, indexLayer = 0 )
        self.genome.add_node(bias)
        # random conection
       
        #self.add_conn_rnd()
        #self.add_conn_rnd()

        self.add_connection2()
        self.add_connection2()

        self.add_node_hidden()
        
        self.specie = ""

        self.champion = False

        self.name = self.new_name()

    def setChampion(self, champ = False):

        self.champion = champ 

    def getChamp(self):
        
        return self.champion

    def set_specie( self, specie_new):

        self.specie = specie_new

    def re_add(self):
        #self.tf_session = tf.Session()
        self.champion = False
        
        for gene in self.genome.nodes_iter():
            
            gene.set_added_to_graph(added = False )
       
            
        
    def add_gene_rnd(self , gene):
        # agregar gene en una position aleatoria
        ## MUTACION FIGURA 3 B, Paper neat
        self.inovationNumber = self.inovationNumber + 1 
        edge = self.get_rand_edge()
        #print(edge)
        weight_1 = self.getRnd()
        weight_2 = self.getRnd()
        
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

    def add_connection2(self):
        """
        ADD A CONEXTION BETWEEN AN EXISTING NODE AT LAYER K, TO A NODE IN LAYER K+1
        
        """
        k = random.randint( 0 , self.HLayers ) # 0 1
       
        gene_k =  self.get_rand_gene( indexLayer = k )
        gene_k_1 = self.get_rand_gene( indexLayer = k +1 )

        if not gene_k :
            # no hay gene en la capa k, agregar uno
            gene_k =  Gene(0.0,graph = self.graph, indexLayer = k )
            self.genome.add_node(gene_k)

        if not gene_k_1 :
            gene_k_1 = Gene(0.0,graph = self.graph, indexLayer = k+1 )
            self.genome.add_node(gene_k_1)

        weight = self.getRnd()
        
        self.add_genes(gene_k , gene_k_1 , weight)

    def add_node_hidden(self):

        # agregar un nodo a una capa oculta
        # y conectarlo
        # capa oculta aleatoria
        hiddenLayer = random.randint( 1 , self.HLayers )

        nextLayer = hiddenLayer + 1
        prevLayer = hiddenLayer -1

        genePrev = self.get_rand_gene(indexLayer = prevLayer)
        geneNext = self.get_rand_gene(indexLayer = nextLayer )
        

        if not genePrev:
            genePrev = Gene(0.0,graph = self.graph, indexLayer = prevLayer )
            self.genome.add_node(genePrev)
        if not geneNext:
            geneNext = Gene(0.0,graph = self.graph, indexLayer = nextLayer )
            self.genome.add_node( geneNext )

        geneHidden = Gene(0.0,graph = self.graph, indexLayer = hiddenLayer )

        w1 = self.getRnd()
        self.add_genes( genePrev , geneHidden ,  w1 )
        w1 = self.getRnd()
        self.add_genes( geneHidden , geneNext , w1)
            
        
    
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

        

    def get_rand_edge(self):

        edge = random.sample( self.genome.edges() , 1)
        return edge
    
    def get_rand_gene(self,  indexLayer = 0 ):
        layerGenes = [] 
        for gene in self.genome.nodes_iter():
            if gene.getIndexLayer() == indexLayer:
                layerGenes.append(gene)

        if not layerGenes:
            return None
        
        gene = random.choice(layerGenes)
        return gene # random gene in de layer indexLayer

    
    def add_gene(self, gn):
        #gn = Gene(0.0)
        self.genome.add_node(gn)
        
  
    def build_graph(self, session = None  , graph_n = None):
        
        if not graph_n:
            print("PLEASE FEED SOME GRAPH")
            return True
        
        self.graph = graph_n
        for gene in  self.genome.nodes_iter():

            if gene.isAdded():
                
                gene.set_added_to_graph(added = False)

        
        for gene in self.outputGenes:
            #gene.activation2( graph = graph_n )
            #gene.set_added_to_graph(True)
            self.activate_predecessors( gene  , graph_n = self.graph )
            #print("OUTPUT NODE ADDED TO GRAPH ")
            gene.set_added_to_graph(added = True)
            #gene.activation2(graph = graph_n )

        for gene in self.inputGenes:

            if  gene.act  == None or not gene.isAdded() :
                with graph_n.as_default():
                    gene.act = tf.placeholder( tf.float32 , shape = () )
                    gene.set_added_to_graph(added = True )
                    #print("RE ADDING PERRRO ")
                    #assert gene.act.graph is graph_n 
                    #print( assert )

    def activate_predecessors(self , gene , graph_n = None ):
        # change to active predecessors
        
        ws = []
        xs = []
        # print( len( self.genome.predecessors(gene)  ) )
        prede_zozios = ( n for n in self.genome.predecessors(gene) )
        for pred in prede_zozios:
            # pred es nodo precedesor
            # por cada nodo precedessor debemos colectar la operacion
            # peso*ops nodo
           
            #print( pred.isAdded() )
            if not pred.isAdded():
                
               
                #ws.append( self.genome[pred][gene]['weight'] )
                
                self.activate_predecessors( pred  , graph_n = self.graph  )
                
                pred.set_added_to_graph(added = True)
                #pred.activation2( graph = graph )
                w = self.genome[pred][gene]['weight']
                
                
                ws.append( self.genome[pred][gene]['weight'] )
                xs.append( pred.act  )
                
                
    
        with graph_n.as_default():

            if gene.bias_gene:

                 gene.act = tf.Variable( tf.constant(1.0 , shape=[] , dtype=tf.float32) , dtype = tf.float32  )
                 gene.set_added_to_graph(added = True )
            
            if gene.input_gene:
               # print("PLACE HOLDERssssssss")
                gene.act = tf.placeholder( tf.float32 , shape = () )
                #print( gene.act )
                gene.set_added_to_graph(added = True )
            else:

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
        
        #print("Respuesta del grafo")
        if not inputs:
            return False

        placeholders = self.get_list_placeholders()
        
        out = []
        out_val = []
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            #print(init)
            session.run(init)
      

       
        resp = 0
       
        
        #with session  as sess:
        for node in self.outputGenes:
                #sess.run( init , feed_dict =   {} )
            if node.act == None:
                print("this happen")
                continue
            
            with self.graph.as_default():
                a = session.run( node.act , feed_dict  = { i : d for i,d in zip( placeholders ,inputs) } )
                out_val.append( a) 
                out.append( node.act )

            

        if len( out ) == 0:
            # ningun nodo de salida dio respuesta
            
            print (" OUT NODEs GIVE NO  ANWSER")
            return False
        #print(out_val) 
        if get_index:
            #with session as sess:
            #out = tf.pack( out , axis = 0)
            #out = tf.argmax(out , axis = 0 )
            
            #resp = int (  session.run( out ) )
            # la respuesta va a ser el indice del de mayor valor
            resp =  [  out_val.index(max( out_val ) ) ] 
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
        self.inputGenes = []
        self.outputGenes = [] 
        self.genome.clear()

    def add_edges_from(self, nedges):
        
        self.genome.add_edges_from(  nedges )

    def add_nodes_from(self , nnodes , is_input = False , is_output = False):

        
        self.genome.add_nodes_from(nnodes)
        
        if is_input:
            self.inputGenes = nnodes[:]

        if is_output :
            self.outputGenes = nnodes[:]

            


    def mutate_weights(self):
        
        for u , vs in self.get_edge_iter():
            for v , atts in vs.items():

                rnd1 = random.random()
                if rnd1 <= 0.9:
                    # muta por perturbacion
                   
                    pertb = self.getRnd()
                    new_weight = ( atts['weight'] + ( pertb*0.5 ) )
                   
                    self.genome[u][v]['weight'] = new_weight
                    continue
                rnd1 = random.random()
                if rnd1 <= 0.1 :
                    # nuevo peso aleatorio 
                    new_weight = self.getRnd()
                    self.genome[u][v]['weight'] = new_weight

                
    def mutate_perturbation(self):
        #   DEPRECATED 
        #do something
        rnd1 = random.random()
        #rnd2 = random.random()
        
        for u , vs in self.get_edge_iter():
            for v , atts in vs.items():
                 
                new_weight = ( atts['weight'] + ( rnd1*0.5 ) )
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
        
        #self.add_conn_rnd()
        self.add_connection2()
        return 0.0

    def mutate_new_node(self):

        #Gnew = Gene( 0.0 , graph = self.graph)
        #self.add_gene_rnd( Gnew )
        self.add_node_hidden()
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


    def getRnd(self):

        return random.uniform(-1 , 1 )
    
    def Draw(self):

        nx.draw(self.genome  )
        plt.show()


    
    def new_name(self, size=6, chars=string.ascii_uppercase + string.digits):
        
        return ''.join(random.choice(chars) for _ in range(size)  )

