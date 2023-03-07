//  Author: Rohan Singh
//  Feb 17, 2023

//  Imports
#include <iostream>

using namespace std;

#define MAX 100


//  Class for Neural Networks
class NeuralNetwork{
    
    // Instance Fields
    private:


    double* weights;
    int dim;
    double epsilon;
    int max;
    double* target;

    // Methods
    public:
    
    NeuralNetwork(int dimensions, double eps, int max_elm, double outputs[]){
        this->dim = dimensions;
        weights = new double[this->dim];
        this->epsilon = eps;
        this->max = max_elm;
        this->target = outputs;

    }

    double get_output(double vec[]){
        double output = 0;
        for(int i = 0; i < dim; i++){
            output+=(this->weights[i])*(vec[i]);
        }
        return output;
    }

    double* get_NN_output(double *vec[]){

        double* out= 0;

        for(int i = 0; i < this->max; i++){
            *(out+i) = get_output(vec[i]);
        }

        return out;
    }

    int get_dimensions(){
        return this->dim;
    }

    double* get_weights(){
        return this->weights;
    }

    void print_weights(){
        for(int i = 0; i < this->dim; i++){
            cout<<this->weights[i];
            cout<<" ";
        }
        cout<<"\n";
    }

    void update_weights(double gradient[]){
        for(int i = 0; i < this->dim; i++)
            this->weights[i]-=(this->epsilon*gradient[i]);
    }

    void set_weights(double* temp_weight){
        this->weights = temp_weight;
    }

    double get_TSE(double output[]){
        double tse= 0;

        for(int i = 0; i < this->max; i++){
            tse+=((output[i]-target[i])*(output[i]-target[i]));
        }

        return tse/2;
    }


};


//  Testing method
void test();


int main(){

    test();
    
    return 0;
}


void test(){

    double target[] = {0,0,0,0,0};

    NeuralNetwork* NN = new NeuralNetwork(10,0.1,20,target);

    NN->print_weights();

    double temp[] = {1.0,1.0,1.0,2.0,3.0,1.0,1.0,1.0,2.0,3.0};

    NN->set_weights(temp);

    NN->print_weights();

    double input[] = {1.0,1.0,1.0,2.0,3.0,1.0,1.0,1.0,2.0,3.0};

    cout<<NN->get_output(input);

    cout<<"\n";

    NN->update_weights(input);

    NN->print_weights();

}