# ES654-2020 Assignment 3

*Your Name* - *Your Roll Number*

------

## Run time comparision of Normal and Gradiant descent Method

1. Run time for different values of N and P for Normal method:
    > For varying P: 

       Time for N=1000 and P=20 is : 0.002765178680419922
       Time for N=1000 and P=40 is : 0.003993511199951172
       Time for N=1000 and P=60 is : 0.009785890579223633

    > For varying N:

        Time for N=1000 and P=50 is : 0.008004426956176758
        Time for N=10000 and P=50 is : 0.015992164611816406
        Time for N=100000 and P=50 is : 0.10430788993835449

2. Run time for Gradiant descent method: 

    > For N=10000 and P=50 and batch_size=10000 is:

        Time for N=10000 , P=50 , iteration=20 is : 0.015644073486328125
        Time for N=10000 , P=50 , iteration=40 is : 0.04021477699279785
        Time for N=10000 , P=50 , iteration=60 is : 0.048037052154541016
        Time for N=10000 , P=50 , iteration=80 is : 0.07306575775146484
        Time for N=10000 , P=50 , iteration=100 is : 0.08013081550598145 

    > For N=10000 and P=50 and batch size=100 is:

        Time for N=10000 , P=50 , iteration=20 is : 0.0
        Time for N=10000 , P=50 , iteration=40 is : 0.007997751235961914
        Time for N=10000 , P=50 , iteration=60 is : 0.008000850677490234
        Time for N=10000 , P=50 , iteration=80 is : 0.008013486862182617
        Time for N=10000 , P=50 , iteration=100 is : 0.008007049560546875

Now we can compare Run time of these two methods.

From first time sample we can conclude that run time for Normal is proportional to (D^2) and (N). 

Where as for Runtime for Gradiant is proportional to (t) i.e., number of iterations

We can observe that gradiant method is better than Normal with either small iterations or small batch size
