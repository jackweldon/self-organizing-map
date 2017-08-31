# self-organizing-map

After first creating a serial version in C I then parallelized certain sections of code using CUDA to speed up the organization.

This program has nodes that are assigned a random colour. A node is selected at random, then a radius of neighbors is selected and using a euclidean distance algorithm there colour weight is changed. Dependent on how close a node is to the randomly selected node its colour is changed more heavily. Eventually after X iterations it organises itself into colours.

An example can be seen here
 https://www.youtube.com/watch?v=j2f0OtcQWaM
