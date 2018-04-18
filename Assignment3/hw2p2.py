import numpy as np
import sys

def sigmoid(z):
    y = 1 / (1.0 + np.exp(-z))
    return np.clip(y,1e-13,1-(1e-13)) #np.clip to avoid value overflow

def cal_accurancy(data, w, b):
    # count error with training set
    train_X = np.array(data[:, :-1], dtype='float')
    train_y = np.array(data[:, -1:], dtype='float').ravel()
    predict = np.dot(train_X,w) + b
    predict = sigmoid(predict)
    
    cross_entropy = -(np.dot(train_y, np.log(predict)) + np.dot((1-train_y), np.log(1-predict)))
    classification = lambda i: 1 if i >= 0.5 else 0
    predict = [classification(i) for i in predict]
    
    err_times = 0
    for i in range(train_y.shape[0]):
        if predict[i] == train_y[i]:
            err_times += 1
    print(err_times / train_y.shape[0], cross_entropy)


def logistic_regression_SGD(data, learning_rate = 1, epochs = 500):
    batch_size = 25
    # assume w and b with all ones will have nuch performance than zeros
    w = np.zeros((106), dtype='float')
    b = np.zeros((1), dtype='float')
    #adagrad strategy
    pre_w = np.zeros((106), dtype='float')
    pre_b = np.zeros((1), dtype='float')
    
    z = np.ones((batch_size), dtype='float')
    y = np.ones((batch_size), dtype='float')
    
    print('training start..')
    for _ in range(epochs):
        batch_iteration = 0
        np.random.shuffle(data) # shuffle data to avoid SGD converge at local minimum
        # an epoch
        #in default, a epoch run 32561 / 25 = 1302... batch times
        while batch_iteration * batch_size < (data.shape[0]): 
            #data extraction
            batch_data = data[batch_iteration*batch_size : batch_iteration*batch_size+25]
            train_X_batch = np.array(batch_data[:, :-1],dtype='float')
            train_y_batch = np.array(batch_data[:, -1:],dtype='float').ravel() #change dimension to 1-D array
            #print('train_X_batch.shape =',train_X_batch.shape,'\n'+'w.shape =',w.shape,'\n'+'b.shape =',b.shape)
            #print('z.shape =',z.shape,'\n'+'y.shape =',y.shape,'\n'+'train_y_batch.shape =',train_y_batch.shape)

            #predict
            z = np.dot(train_X_batch,w) + b
            y = sigmoid(z)

            # update weights and bias
            # use mean instead sum will imporve performance
            w_gradient = np.mean(-1 * train_X_batch * (train_y_batch-y).reshape(batch_data.shape[0],1), axis = 0)
            b_gradient = np.mean(-1 * (train_y_batch-y))
            pre_w += w_gradient**2
            pre_b += b_gradient**2
            ada_w = np.sqrt(pre_w)
            ada_b = np.sqrt(pre_b)

            w = w - learning_rate * w_gradient / ada_w
            b = b - learning_rate * b_gradient /ada_b
            batch_iteration += 1
            
    cal_accurancy(data, w, b)
    # save model
    w.tofile('./w.model')
    b.tofile('./b.model')
    print('training finish..')

            
def training_model(x_filename = './X_train', y_filename = './Y_train'):
    print("Loading dataset..")
    with open(x_filename) as X_f, open(y_filename)as y_f:
        data = []
        while True:
            train_X = X_f.readline()
            if(not train_X): #read until end of file
                break
            single_example = []
            train_X = train_X.strip().split(',') # pre-process with '\n' and ','
            train_y = y_f.readline().strip().split(',')  # pre-process with '\n' and ','
            single_example.extend(train_X)
            single_example.extend(train_y)
            data.append(single_example)
        data = np.array(data[1:]) # ignore row of attribute name
        # print(data.shape) 

        # Standardization
        train_X = np.array(data[:, :-1], dtype='float')
        train_y = np.array(data[:, -1:], dtype='float')

        std = train_X.std(axis=0)
        mean = train_X.mean(axis=0)
        train_X = (train_X - mean) / std
        
        data = np.hstack((train_X, train_y))
        std.tofile('./stdandardize_std.model')
        mean.tofile('./stdandardize_mean.model')
        
        #training    
        logistic_regression_SGD(data, learning_rate = 1, epochs = 500)

def testing(filename = './X_test'):
    print('Starting testing..')
    #read testing dataset
    with open(filename) as f:
        test_data = []
        test_num = 0
        while True:
            test_X = f.readline()
            if(not test_X): #read until end of file
                break
            test_num += 1
            single_example = []
            test_X = test_X.strip().split(',') #pre-process with '\n' and ','
            single_example.extend(test_X)
            test_data.append(single_example)
        test_data = np.array(test_data[1:]).astype('float') #ignore row of attribute name
        
        #loading model
        w = np.fromfile('./w.model')
        b = np.fromfile('./b.model')
        std = np.fromfile('./stdandardize_std.model')
        mean = np.fromfile('./stdandardize_mean.model')
        test_data = (test_data - mean) / std #Standardization

        #predict
        z = np.dot(test_data,w) + b
        y = sigmoid(z)
        
        #write to file
        write_to_file(y)
    print('Testing finishing..')

def write_to_file(test_y,filename = './predictions.csv'):
    with open(filename, 'w') as f:
        classification = lambda i: 1 if i >= 0.5 else 0
        predict = [classification(i) for i in test_y]
        f.write('id,label\n')
        for i in range(len(predict)):
            f.write(str(i) + ',' + str(predict[i]) + '\n')
            #print(test_y[i])

        
if __name__ == '__main__':
    # for training
    # python hw2p2.py --mode train --file X_train Y_train
    # for testing
    # python hw2p2.py --mode test --file X_test
    
    if len(sys.argv) != 1:
        #print(sys.argv[2],sys)
        if sys.argv[2] == 'train':
            training_model(x_filename = sys.argv[4], y_filename = sys.argv[5])
        elif sys.argv[2] == 'test':
            testing(sys.argv[4])
    else:
            training_model()
            testing()       
    

