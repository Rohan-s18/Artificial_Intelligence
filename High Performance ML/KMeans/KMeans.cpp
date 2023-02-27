//  Author: Rohan Singh
//  Feb 27, 2023
//  Code for high performance KMeans Clustering algorithm in C++

//  Imports
#include <iostream>
#include <cmath>
#include "Testing.hpp"
#include <float.h>


//  Class for KMeans
class KMeans{

    //Helper functiona and instance fields
    private:

    //Instance fields
    int dim;
    int data_len;
    int k;
    double** dataset;
    int* cluster;
    double** centroids;
    int* cluster_len;


    //Helper functions
    
    //function that returns the Euclidean distance
    double euc_dist(double x[], double y[]){
        double dist = 0;
        for(int i = 0; i < dim; i++)
            dist+=((x[i] - y[i])*(x[i] - y[i]));
        return sqrt(dist);
    }

    //function for the objective function of KMeans algorithm
    void objective_function(){

        //Iterating though all of the datapoints
        for(int i = 0; i < this->data_len; i++){
            int cl = -1;
            double dist = DBL_MAX;
            double temp = 0;

            //Iterating through all of the centroids
            for(int j = 0; j < this->k; j++){
                temp = this->euc_dist(this->dataset[i],this->centroids[j]);

                if(temp < dist){
                    dist = temp;
                    cl = j;
                }
            }

            //Classifying the cluster for 'i' as cl
            this->cluster[i] = cl;

        }

    }


    //function to update the centroids for each cluster
    void update_centroids(){

        //Creating matrix for new centroids
        double** updated_centroids = new double*[this->k];
        for(int i = 0; i < this->k; i++)
            updated_centroids[i] = new double[this->dim];
        


        //Iterating through each data point
        for(int i = 0; i < this->data_len; i++){

            //Holds the sum for each dimension
            double cluster_sum[this->dim];
            
            //Adding the 

        }







    }


    //Public functions
    public:

    KMeans(int dimensions, int data_length, int num_clusters, double* data[]){
        this->dim = dimensions;
        this->data_len = data_length;
        this->k = num_clusters;
        this->dataset = data;
        this->cluster = new int[data_length];
        this->cluster_len = new int[num_clusters];
        this->centroids = new double*[num_clusters];
        for(int i = 0; i < num_clusters; i++)
            this->centroids[i] = new double[this->dim];

    }

    //Predict function which returns the cluster for each corressponding data point
    int* predict(){

        return this->cluster;
    }

    void print_centroids(){
        for(int i = 0; i < this->k; i++){
            std::cout<<"[";
            for(int j = 0; j < this->dim; j++)
                std::cout<<this->centroids[i][j]<<" ";
            std::cout<<"]\n";
        }
    }



};


//  Main function
int main(){

    kmeans_demo();

    return 0;
}


//  Implementation of demo
void kmeans_demo(){

    KMeans* demo = new KMeans(5, 100, 6, NULL);

    demo->print_centroids();

}

