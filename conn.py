
import numpy as np

from gene import Gene
# connection between genes
class Conn():

    def __init__(self, source_gene , weight ):

        self.source_gene = source_gene
        self.weight = weight

    def act_source(self):

        return self.source_gene.act

    def get_weight(self):

        return self.weight

    



        
