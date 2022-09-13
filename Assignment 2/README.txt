###################################################################
DATA: house dataset, change to desired directory and provided PointViewMatrix.py

The code in fundamental_matrix.py estimates fundamental matrices using different methods of the 8-point
algorithm and visualizes the epipolar lines. At the beginning of the file one can select a specific flag
which will determine which method will be employed, i.e. regular 8-point, its normalized variant or its 
normalized variant with RANSAC.   

The code in create_PVM.py constructs a custom sparse PVM. 

The code in structure-from-motion.py contains the structure from motion algorithm. This is employed on both the
provided PVM PointViewMatrix.txt and for the PVM created by create_PVM.py. Different frame rates can be given
as input. In the function "motion_structure()" one can specify if ambiguity removal should be employed.
#######################################################################################################