"""
Author: Rohan Singh
Python Script for Testing
"""

# imports
import numpy as np
from unit import Unit


# test suite for unit
def unit_test():
    data = [1,34,5,-9,10,11,0,32,-15,-6,-11]

    test_unit = Unit(11,4,"SIGMOID")

    print("\nFIRST PREDICTION")
    print(test_unit.forward(data=data))

    test_unit.set_weights(np.ones(11)*-1)

    print("\nSECOND PREDICTION")
    print(test_unit.forward(data=data))


def main():
    unit_test()


if __name__ == "__main__":
    main()