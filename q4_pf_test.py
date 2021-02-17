import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures
X = np.array([1,2])
polyb = PolynomialFeatures(3)
polyn=PolynomialFeatures(3,False)
print("Transform with bias: ")
print(polyb.transform(X))
print()
print("Transform without bias: ")
print(polyn.transform(X))
