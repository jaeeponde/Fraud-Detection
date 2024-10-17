import pandas as pd
import numpy as np
import pickle
import argparse
from data_preprocessing import preprocess

#class node is to store every node in the tree along with its features. left, right are to store children. feature denotes which feature this node has been split at. 
class Node():
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        """
        self.feature = feature # which feature this node is split a
        self.threshold = threshold # minimum info gain for split
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value # decision made by this node       

#calculates all possible splits with impurity measure and best split 
class DecisionTree():
    

    def __init__(self, min_samples=2, max_depth=2):
     
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
    
        left_dataset = []
        right_dataset = []
        
        
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y): 
        
        entropy = 0

        
        labels = np.unique(y)
        for label in labels:
            
            label_examples = y[y == label]
            
            pl = len(label_examples) / len(y)
            
            entropy += -pl * np.log2(pl)

        
        return entropy

    def information_gain(self, parent, left, right):

        #info gain = entropy parent - sum of entropy of children
    
        information_gain = 0
       
        parent_entropy = self.entropy(parent)
        
        samples_left = len(left) / len(parent)
        samples_right= len(right) / len(parent)
        
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        
        weighted_entropy = samples_left * entropy_left + samples_right * entropy_right
        
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset, num_samples, num_features):
        
        #chooses best split as the one with maximum information gain 
        
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        
        for feature_index in range(num_features):
            
            feature_values = dataset[:, feature_index]
            
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                
                if len(left_dataset) and len(right_dataset):
                    
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    
                    information_gain = self.information_gain(y, left_y, right_y)
                    
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split

    
    def calculate_leaf_value(self, y):
      
        y = list(y)
        
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth=0):
        
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            
            best_split = self.best_split(dataset, n_samples, n_features)
            
            if best_split["gain"]:
                
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
               
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        
        leaf_value = self.calculate_leaf_value(y)
       
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)

    def predict(self, X):
        
        predictions = []
       
        for x in X:
            prediction = self.make_prediction(x, self.root)
            
            predictions.append(prediction)
         
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        
        if node.value != None: 
            return node.value
        else:
           
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
            

#following functions are for metric predictions 

def confusion_matrix_np(y_true, y_pred):
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TP = np.sum((y_pred == 1) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

def accuracy_score_np(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score_np(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall_score_np(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score_np(y_true, y_pred):
    precision = precision_score_np(y_true, y_pred)
    recall = recall_score_np(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix_np(y_true, y_pred)
    accuracy = accuracy_score_np(y_true, y_pred)
    precision = precision_score_np(y_true, y_pred)
    recall = recall_score_np(y_true, y_pred)
    f1 = f1_score_np(y_true, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

def save_metrics(metrics_output_path, accuracy, precision, recall, f1, conf_matrix):
    
    with open(metrics_output_path, 'w') as metrics_file:
        metrics_file.write(f"Classification Metrics:\n")
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n")
        metrics_file.write(f"Precision: {precision:.4f}\n")
        metrics_file.write(f"Recall: {recall:.4f}\n")
        metrics_file.write(f"F1-Score: {f1:.4f}\n")
        metrics_file.write(f"Confusion Matrix:\n")
        metrics_file.write(f"[[{conf_matrix[0, 0]}, {conf_matrix[0, 1]}],\n")
        metrics_file.write(f" [{conf_matrix[1, 0]}, {conf_matrix[1, 1]}]]\n")

def save_predictions(predictions_output_path, predictions):
    
    pd.DataFrame(predictions, columns=['Predictions']).to_csv(predictions_output_path, index=False, header=False)

def main():
    #takes in the arguments from the bash command and uses them for prediction
    parser = argparse.ArgumentParser(description='Load model, make predictions, and output metrics and predictions.')
    parser.add_argument('--model_path', required=True, help='This is where the model is stored')
    parser.add_argument('--data_path', required=True, help='Data that has to be tested')
    parser.add_argument('--metrics_output_path', required=True, help='Output accuracy, F1, recall, precision, confusion matrix')
    parser.add_argument('--predictions_output_path', required=True, help='Path to save the predictions output file')
    
    args = parser.parse_args()

    
    with open(args.model_path, 'rb') as model_file:
        model = pickle.load(model_file)

   
    data = preprocess(args.data_path)

    #unpickles the model and makes predictions 
    X = data.drop(columns=['isFraud']).values   
    y_true = data['isFraud'].values.reshape(-1,1)
    
    y_pred = model.predict(X)

    y_pred = np.round(y_pred).astype(int)
    y_true = y_true.flatten()

    
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(y_true, y_pred)

   
    save_metrics(args.metrics_output_path, accuracy, precision, recall, f1, conf_matrix)

    
    save_predictions(args.predictions_output_path, y_pred)

if __name__ == '__main__':
    main()

