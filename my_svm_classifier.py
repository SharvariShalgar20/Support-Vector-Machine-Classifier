import numpy as np

class SVM_Classifier:

  #Initializing the hyperparameter
  def __init__(self,learning_rate, no_of_iteration, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration
    self.lambda_parameter = lambda_parameter


  #fitting the dataset to SVM Classifier
  def fit(self, X, y):
    #m is total number of datapoints(number of rows)
    #n is total number of feature(number of column)
    self.m,self.n = X.shape

    #initiating the weight and bias value
    self.w = np.zeros(self.n)        #w will be a array where total elements are the input columns are initial they are zeros

    self.b = 0
    self.X = X
    self.y = y

    #implementing gradient descent algorithm for optimization
    for i in range(self.no_of_iteration):
      self.update_weights()



  #function for updating the weight and bias value
  def update_weights(self):
    #label encoding
    y_label = np.where(self.y <= 0, -1, 1)   #this mean if the outcome is 0 then convert it to -1 otherwise leave 1 as it is

    #Gradient(dw, db)
    for index, x_i in enumerate(self.X):
      #x_i are the all the values of X(features) correspondence to index
      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

      if(condition == True):
        dw = 2 * self.lambda_parameter * self.w
        db = 0

      else:
        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
        db = y_label[index]

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db



  #predict the label for a given input value
  def predict(self, X):
    
    output = np.dot(X, self.w) - self.b
    predicted_labels = np.sign(output)  #(yi-cap)

    #converting all -1 to zero back again
    y_cap = np.where(predicted_labels <= -1, 0, 1)

    return y_cap
