The aim of this project, is construct an evolutional neural network capable of learning to play a puzzle game called 2048


./main_test.py this is the main file, here the training method is called over the population with the goal to train

./population.py It contains the logic of the genome population. It stores each one of the genes, and also cointains the object that mediates the crossover process. Its most important methods, are the mutate() which is in charge of mutate the genes according to some rules ( mutating weigths or mutating the topology ), also organizing the genome species is important to the neuroevolution process and to make more rich the population, in terms of diversity. The reproducction process is also handled by this class

./scrapper.py It contains the code needed to get the score count for the game from the web page. And also contains the code necessary to send keystrokes from the neural network output to the web game.

./train.py Contains the logic to connect the population to the web page that hosts the game. It 

./capture.py Module for screen capture and all the preprocessing of the image, and information needing for training.

./image_processing.py It contains the code necessary to transform the image captured from the web, to an array of ints, this ints are going to be the input for the neural network. This code is heavily based on opencv.

./gene.py this contains the gene class. a gene is the fundamental unit, a set of genes and their connections make a genome.

./genome.py the class genome is a collection of genes and the connections and weights. each genome is a neural network, and the evolution process is based on gradually improve the performance of the genes in a population for a determinined task.In the case, getting a high score for the game.


After a while of experimentation and testing, i came to realize that this may not be the better way to do it. The Tensorflow graphs arent thinked to change over time in topology and everything gone here is a huge work around.




