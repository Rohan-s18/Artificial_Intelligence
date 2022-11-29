from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Using Sci-kit learns multi layer perceptron on the iris dataset for classification

def run():
    #print("Hello World!")
    #I had a problem with pandas dataframes while using MLP, so I used scikit's load_iris() as the format worked
    iris = load_iris()

    #Splitting the data into test and training sets
    datasets = train_test_split(iris.data, iris.target, test_size=0.3)

    train_data, test_data, train_labels, test_labels = datasets

    scaler = StandardScaler()

    # Fititng the train data
    scaler.fit(train_data)

    # scaling the train data
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    #Training the model
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

    #Fiiting the training model
    mlp.fit(train_data, train_labels)

    #Running the prediction function on the training data
    predictions_train = mlp.predict(train_data)
    
    #Running the prediction function on the testing data
    predictions_test = mlp.predict(test_data)

    """
    print(accuracy_score(predictions_train, train_labels))
    print(accuracy_score(predictions_test, test_labels))
    print(classification_report(predictions_test, test_labels))
    """

    return predictions_test, predictions_train

def main():
    iris = load_iris()
    print(type(iris.data))
    test_results, train_results = run()
    print(test_results)
    print(train_results)

if __name__ == "__main__":
    main()