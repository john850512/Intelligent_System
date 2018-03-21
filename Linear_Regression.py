import csv
import time
import ast
import numpy as np
import sys

def type_converter(string_list): #change str to correct type
    coverted_list = []
    for i in string_list:
        try:
            coverted_list.append(ast.literal_eval(i))
        except:
            coverted_list.append(i) #some still with str type (ex:date、chinese)
    return coverted_list
            
def null_converter(_list): # parameter _list must be a str list
    return [w.replace("NR","0") for w in _list]

def train_model(filename = './train.csv'):
    with open(filename) as f:  
        ### load data ###
        data = [[] for i in range(18)]
        row_idx = 0
        fir_flag = True
        for row in csv.reader(f): #將逗號、""刪除
            if fir_flag == True:
                fir_flag = False
                continue
            row = null_converter(row)
            #row = type_converter(row)
            data[row_idx % 18].extend(row[3:])
            row_idx += 1
        data = np.array(data)
        print(filename.strip('./'),"loading finish...data dimensions are:",data.shape,',data count:',row_idx)
        
        ### feature extraction ###
        train_x = np.zeros(shape=(5652,163)) # 471 * 12 = 5652, 18 * 9 + 1 = 163
        train_y = np.zeros(shape=(5652,1)) # 471 * 12 = 5652
        count_instance = 0
        for month in range(12): # 每個月20 * 24 - 10 + 1 = 471筆
            shift = 0
            idx_x = 0
            idx_y = 0
            count_attribute = 0
            while shift < 471: #處理完一個月
                if idx_x == 9 and idx_y == 9: #處理第十筆的PM2.5
                    train_y[count_instance][0] =  data[idx_x][month*480 + idx_y + shift]
                    #print('將data[',idx_x,'][',month*480 + idx_y + shift,']:',data[idx_x][month*480 + idx_y + shift],'移到','train_y[',count_instance,'][0]')
                    idx_x += 1
                    idx_y = 0
                    continue
                elif idx_x == 18: #處理完一筆資料(18*9 + 1 = 163筆)
                    train_x[count_instance][count_attribute] = 1 #bias
                    #print('將bias:1新增到','train_x[',count_instance,'][',count_attribute,']')
                    count_instance += 1
                    shift += 1
                    count_attribute = 0
                    idx_x = 0
                    #s = input()
                    continue
                elif idx_y == 9: #處理完一筆資料中的一個屬性
                    idx_x += 1
                    idx_y = 0
                    continue
                else:
                    train_x[count_instance][count_attribute] = data[idx_x][month*480 + idx_y + shift]
                    #print('將data[',idx_x,'][',month*480 + idx_y + shift,']:',data[idx_x][month*480 + idx_y + shift],'移到','train_x[',count_instance,'][',count_attribute,']')
                    count_attribute += 1
                idx_y += 1
        print("feature extraction finish...")
        ### linear regression ###
        np.set_printoptions(threshold=np.inf)
        weight_vector = np.zeros(shape=(163,1))
        learning_rate = 0.3
        pre_gradient = np.zeros(shape=(163,1)) #implement adagrad
        epochs = 5000
        history = []
        
        for _ in range(epochs):
            #print(weight_vector)
            #input()
            
            temp_y = np.dot(train_x,weight_vector)
            #print(temp_y)
            #input()
            loss = temp_y - train_y
            #print(loss)
            #input()
            train_x_transpose = np.transpose(train_x)
            gradient = 2 * np.dot(train_x_transpose,loss)
            pre_gradient += gradient**2
            adagrad = np.sqrt(pre_gradient)
            #print(gradient)
            #input()
            weight_vector = weight_vector - learning_rate * gradient / adagrad
        print('training finish...')  
        return weight_vector

    
def test_model(filename = './sample-test.csv'):
    ### testing ###
    with open(filename) as f:
        
        ### load data ###
        data = [[] for i in range(18)]
        row_idx = 0
        test_data_number = 0
        for row in csv.reader(f): #將逗號、""刪除
            row = null_converter(row)
            #row = type_converter(row)
            data[row_idx % 18].extend(row[2:]) #test資料中沒有日期 所以從2開始
            row_idx += 1
        data = np.array(data)
            
        test_data_number = int(row_idx / 18)
        print(filename.strip('./'),"loading finish...data dimensions are:",data.shape,',data count:',test_data_number)
            
        ### feature extraction ###
        test_x = np.zeros(shape=(test_data_number,163)) 
        count_instance = 0
        count_attribute = 0
        for count_instance in range(test_data_number):
            idx_x = 0
            idx_y = 0
            while True:
                if idx_x == 18: #處理完一筆資料(18*9 + 1 = 163筆)
                    test_x[count_instance][count_attribute] = 1 #bias
                    #print('將bias:0新增到','test_x[',count_instance,'][',count_attribute,']')
                    count_instance += 1
                    count_attribute = 0
                    idx_x = 0
                    break
                elif idx_y == 9: #處理完一筆資料中的一個屬性
                    idx_x += 1
                    idx_y = 0
                    continue
                else:
                    test_x[count_instance][count_attribute] = data[idx_x][count_instance*9 + idx_y]
                    count_attribute += 1
                idx_y += 1
        print("test data feature extraction finish...")
        
        test_y = np.dot(test_x,weight_vector)
        print(test_y)
        print("testing finish...")
        return test_y

def write_to_file(test_y,filename = 'predictions.csv'):
    with open(filename,'w') as f:
        f.write('id,value\n')
        for i in range(len(test_y)):
            f.write('id_'+str(i)+','+str(round(test_y[i],2))+'\n')
    

#np.set_printoptions(threshold=np.inf)
if __name__ == '__main__':
    global weight_vector  
    # python Linear_Regression.py --mode train --file train.csv
    if len(sys.argv) != 1:
        #print(sys.argv[2],sys.argv[4])
        if sys.argv[2] == 'train':
            weight_vector = train_model(filename = sys.argv[4])
            weight_vector.tofile('./data.model')
        elif sys.argv[2] == 'test':
            weight_vector = np.fromfile('./data.model')
            test_y = test_model(filename = sys.argv[4])
            write_to_file(test_y)
    else:
            weight_vector = train_model()
            test_model()  
        
