import numpy as np

if __name__=='__main__':
    label=np.load('./random_ir_train_label.npy')
    num=np.zeros(10)
    for i in range(label.shape[0]):
        num[int(label[i])]+=1
    print(np.max(num)//np.sort(num))