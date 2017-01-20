
import capture as cap
from gene import Gene
from genome import Genome
from crossover import Crossover

import matplotlib.pyplot as plt 

"""
x1 = 10
x2 = 500

y1 = 10
y2 = 500

capturado = cap.Capture( x1 , y1 , x2 , y2)

capturado.captureScreen()
"""

cs = Crossover()
gn1 = Gene( 1.0 , input_gene=True , response = 1 )
gn2 = Gene( 2.0 )

gn3 = Gene( 3.0 )

gn4 = Gene( 5.0 )

genome1 = Genome()
genome2  = Genome()



#genome1.add_genes(gn1 , gn2)
#genome1.add_genes( gn2 , gn3 )
#genome1.add_genes( gn3, gn1 )

"""
genome1.add_gene()
genome1.add_gene()
genome1.add_gene()
genome1.add_gene()


genome1.add_conn_rnd()
genome1.add_conn_rnd()
genome1.add_conn_rnd()
genome1.add_conn_rnd()
genome1.add_conn_rnd()
genome1.add_conn_rnd()
genome1.add_conn_rnd()

genome1.build_graph()

genome1.launch_session()
"""

#genome1.build_xor()
#genome2.build_xor2()

#genome1.build_graph() # tf
#genome1.launch_session()


genome2.update_conns()
genome2.build_graph()
genome2.launch_session()


genome2.Draw()



ggg = cs.combine_genomes( genome1 ,  genome2 )

#print("wtf men")
#genome1.update_conns()
#genome1.build_graph()
#genome1.launch_session()

ggg.Draw()
ggg.update_conns()
ggg.build_graph()
ggg.launch_session()


#ggg.Draw()
