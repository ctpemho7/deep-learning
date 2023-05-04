import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    
    def exp_and_sum(x):
        exp = np.exp(x)
        # суммируем по строкам     
        summ = exp.sum(axis=x.ndim-1)
        return exp, summ
    
    
    # получаем максимум разным способом в зависимости от dim     
    if  predictions.ndim == 1:
        new_predictions = predictions - np.max(predictions)
        exp, summ = exp_and_sum(new_predictions)
        output = exp / summ
        
    else:
        maximum = np.max(predictions, axis=1)
        
        # добавляем новую axis, чтобы избежать ошибки  
        # ValueError: operands could not be broadcast together with shapes (2,3) (2,) 
        new_predictions = predictions - maximum[:, np.newaxis]
        # другой вариант: сделать maximum.reshape((-1, 1))
        # это позволит увел        

        
        exp, summ = exp_and_sum(new_predictions)                
        output = exp / summ[:, np.newaxis]

    return output


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    target_distribution = np.zeros_like(probs)
    # для одномерного
    if type(target_index) == int:
        target_distribution[target_index] = 1
    else:
        # n-мерное     
        target_distribution[np.arange(len(target_index)), target_index.reshape(-1)] = 1
    
    output = -np.sum(target_distribution * np.log(probs)) # / probs.shape[0] 
    
    return output
    
    
def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    target_distribution = np.zeros_like(predictions)
    # для одномерного
    if type(target_index) == int:
        target_distribution[target_index] = 1
    else:
        # n-мерное     
        target_distribution[np.arange(len(target_index)), target_index.reshape(-1)] = 1
    
    dprediction = softmax(predictions)
    
    loss = cross_entropy_loss(dprediction, target_index)     
    dprediction[target_distribution.astype(bool)] = dprediction[target_distribution.astype(bool)]-1
    
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # l2_reg_loss = reg_strength * sumij W[i, j]2
    
    loss = reg_strength * np.sum(W ** 2)
    grad = reg_strength*2*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpredictions)
        
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for idx in batches_indices:
                batch_X = X[idx]            
                batch_y = y[idx] 
                # считаем градиент и регуляризацию             
                loss, dW = linear_softmax(batch_X, self.W, batch_y)
                l2_loss, l2_dW = l2_regularization(self.W, reg)
                # лосс
                total_loss = loss + l2_loss
                total_dw = dW + l2_dW
                # обновляем веса
                self.W = self.W - learning_rate*total_dw

                loss_history.append(total_loss)
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # y_pred = np.zeros(X.shape[0], dtype=int)
        pred = np.dot(X, self.W)
        probabilities = softmax(pred)

        y_pred = np.argmax(probabilities, axis=1)

        return y_pred



                
                                                          

            

                
