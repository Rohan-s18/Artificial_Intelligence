//  Author: Rohan Singh
//  Date: Feb 22, 2023
//  Single Variable Linear Regression in C++


//  Imports
#include <iostream>

//  Using header file to get testing/demo code
#include "test.hpp"


//  Class for Linear Regression
class LinearRegression{

    //Section for instance fields and helper function
    private:

    //Linear Regression instance fields
    double bias;
    double weight;

    //Dataset fields
    double* target;
    int dataset_length;
    double* dataset;


    //Helper function to get an array for all of the outputs
    double* get_output(double temp_bias, double temp_weight){
        double arr[dataset_length];

        //Traversing through the input array
        for(int i = 0; i < dataset_length; i++){
            arr[i] = (dataset[i]*temp_weight) + temp_bias;
        }

        return arr;
    }

    //Helper function to get the Total Squared Error for the given weights 
    double get_mse(double output[]){
        double mse = 0;

        for(int i = 0; i < this->dataset_length; i++)
            mse += ((output[i]-this->target[i])*(output[i]-this->target[i]));
        
        mse /= dataset_length;
        return mse/2;
    }




    //Section for public function
    public:

    //Constructor
    LinearRegression(double output[], double data[], int len){
        //Setting the variables
        this->target = output;
        this->dataset = data;
        this->dataset_length = len;

        //Initializing the variables
        this->bias = 0;
        this->weight = 0;
    }



};

//  Main function for demonstraiton
int main(){

    std::cout<<"Hello World!\n\n";

    //Calling the demo function
    single_variable_demo();

    std::cout<<"\n\n";

    //Calling the testing function
    single_variable_test();

    std::cout<<"\n";

    return 0;
}


//  Demonstration function
void single_variable_demo(){

}


//  Testing function
void single_variable_test(){

}
