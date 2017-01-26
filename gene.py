
#classe gen


import numpy as np
import tensorflow as tf




class Gene():
    def __init__(self, ws , input_gene = False, output_gene=False ,response =  None , graph = None):
        # pesos  y funcion de activacion
        if not graph:
            print("QUE PASO EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEee")
            return
        self.graph = graph 
        self.ws = ws
        self.input_gene = input_gene
        self.output_gene = output_gene
        self.connex = []
        self.rnd = np.random.randint(10000)
        self.rnd2 = np.random.randint(100)
        self.act = None

        self.added_to_graph = False 
        """
        with self.graph.as_default():
            
            if self.input_gene:      
                self.act = tf.placeholder( tf.float32 , shape=() )
                print("-") 
                assert self.act.graph is self.graph 
            else :
                self.act = tf.Variable( tf.constant( -1.0 ,shape=[],dtype=tf.float32 )  ,dtype=tf.float32 ).initialized_value()
        
        """

        self.indexLayer = 0
        
        
        
    def activation2(self, graph = None):
        if not graph:
            print("PLEASE FEED GRAPH")
            
        if self.input_gene:
            with graph.as_default():
                
                self.act = tf.placeholder( tf.float32 , shape=() )
                print("wtf men")
                    

        ws = []
        xs = []
        for conn in self.connex:
            if conn.source_gene.act == None:
                continue 
            ws.append(conn.get_weight() )
            xs.append(conn.source_gene.act )

        if not ws  or not xs:
            
            return False
        
       
        with graph.as_default():
            ws = tf.Variable( np.array(ws) , name="ws" ,dtype=tf.float32 ).initialized_value()
            xs = tf.pack( xs ,axis=0  )
            self.act = tf.Variable( tf.sigmoid( tf.reduce_sum( tf.mul(ws,xs)  )) , name="activ" ).initialized_value()
            


    def set_added_to_graph(self, added= False):

        self.added_to_graph = added

    def isAdded(self):

        return self.added_to_graph
    
    def add_conn(self  , conn):
        # por el momento conexion agregar luego gestionar
        self.connex.append(conn)


    def clear_conns(self):

        self.connex = []

        
    def rm_conn(self , gene_source ):
        #remover conexion
        conn = next((x for x in self.connex if x.source_gene == gene_source), None)
        if not conn:
            return False
        self.connex.remove(conn)

    
    def isOutput(self):
        return self.output_gene
    
    def isInput(self):
        return self.input_gene
    
    def __hash__(self):

        return  hash( self.rnd )^hash(repr( self ) ) 

    def __eq__(self, other):
        
        return ( self.__hash__ == other.__hash__  and self.rnd == other.rnd and self.rnd2 == other.rnd2 )

    def __cmp__(self, other):
        # no hay un orden en los genes 
        return ( 0 )
        
