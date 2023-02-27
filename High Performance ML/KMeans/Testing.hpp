//  testing Header file for KMeans Clustering


void kmeans_demo();

void kmeans_test();

void print_array(double arr[], int n){
    std::cout<<"[";
    for(int i = 0; i < n; i++)
        std::cout<<arr[i]<<" ";
    std::cout<<"]\n";
};

void print_array(int arr[], int n){
    std::cout<<"[";
    for(int i = 0; i < n; i++)
        std::cout<<arr[i]<<" ";
    std::cout<<"]\n";
};
