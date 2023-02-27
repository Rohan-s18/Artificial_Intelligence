//  Author: Rohan Singh
//  Feb 27, 2023
//  Code for high performance KMeans Clustering algorithm in C++

//  Imports
#include <iostream>
#include <cmath>
#include "Testing.hpp"


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
    double* centroids;
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

    }


    //function to update the centroids for each cluster
    void update_centroids(){

    }


    //Public functions
    public:

    KMeans(int dimensions, int data_length, int num_clusters, double* data[]){
        this->dim = dimensions;
        this->data_len = data_length;
        this->k = num_clusters;
        this->dataset = data;
        this->cluster = new int[data_length];
        this->centroids = new double[num_clusters];
        this->cluster_len = new int[num_clusters];
    }

    //Predict function which returns the cluster for each corressponding data point
    int* predict(){

    }



};


//  Main function
int main(){

    kmeans_demo();

    return 0;
}


//  Implementation of demo
void kmeans_demo(){

}

