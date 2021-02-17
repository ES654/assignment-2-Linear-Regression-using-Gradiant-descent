# ES654-2020 Assignment 3

*Your Name* - *Your Roll Number*

------

## Creating Multicollinearity data and its effect on Gradiant descent and Noraml method

1. The data set Created is: 

    > 
    |  |  0       |     1    |    3      |
    |--| ---      | ----     | ----      |
    |0 | 1.070842 | 1.140070 | 4.560281  |
    |1 |-0.547114 |-0.244773 | -0.979091 |
    |2 |-0.822682 |-0.250461 | -1.001842 |
    |3 |-0.873521 |-1.507629 | -6.030516 |
    |4 | 0.779568 | 1.010348 | 4.041392  |
    |5 | 1.673637 |-2.354003 | -9.416011 |
    |6 |-0.900470 |-2.681540 |-10.726161 |
    |7 |0.190999  |-0.256133 | -1.024532 |
    |8 |-0.642573 |  0.468811|  1.875245 |
    |9 |-0.670554 |-0.403816 | -1.615265 |

    Here column 1 and 3 are related as Col(3) = 4 * Col(1)

2. Result from Gradiant descent Method is :

    > RMSE :  0.6554187139307434

    > MAE  :  0.5875271534826935

3. Result of Normal method :

    > Error, X transpose * X is not invertable

