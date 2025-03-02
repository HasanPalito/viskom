import numpy as np

class MyNN:
    def __init__(self,input_size,layer_number,layer_size,output_size,learning_rate=0.1):
        self.input_size= input_size
        self.layer=layer_number
        self.layer_size=layer_size
        self.learning_rate=learning_rate
        self.matrices_list=[]
        self.matrices_list.append(np.zeros((input_size,layer_size), dtype=float))
        for i in range(layer_number-2):
            self.matrices_list.append(np.zeros((layer_size,layer_size), dtype=float))
        self.matrices_list.append(np.zeros((layer_size,output_size), dtype=float))
        return 

    def foward(self,input):
        layer_output=[]
        for matrice in self.matrices_list:
            output= input @ matrice
            layer_output.append(output)
            input= output
        return output,layer_output
    
    def err(self,prediction,output):
        length=len(output)
        count=0
        sum = 0
        err=[]
        for out in output:
            error = out-prediction[count]
            rms = (((prediction[count])**2)-out**2)**0.5
            sum = sum + rms
            count = count + 1
            err.append(error)
        MSE = sum/length
        return MSE,err
    
    def backward(self,layer_output,input,error,weight):
        grad_matrices = []
        for i in reversed(range(len(layer_output))):
            if i ==0 :
                grad= input.T @ error
            else :
                grad= layer_output[i-1].T @ error
            grad_matrices.append(grad)
            error = error @ weight[i].T
        grad_matrices.reverse()  
        return grad_matrices  

