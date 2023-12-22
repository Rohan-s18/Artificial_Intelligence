"""
Author: Rohan Singh
Python script to test the non-linearities
"""


from non_linearities import *

def main():

    print("\nTESTING RELU")
    print(relu(10))
    print(relu(-10))
    print(relu(0))

    print("\nTESTING SIGMOID")
    print(sigmoid(0))
    print(sigmoid(10))
    print(sigmoid(-10))

    print("\nTESTING ABSOLUTE VALUE")
    print(abs(0))
    print(abs(10))
    print(abs(-10))


if __name__ == "__main__":
    main()