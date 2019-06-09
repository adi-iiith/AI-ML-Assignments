def make_batch(train, batchsize, no_of_batches) :
    global batched_train
    global batched_train_out
    temp = 0
    for i in range(no_of_batches) :
        batched_train[i] = train[:,temp*batchsize:temp*batchsize + batchsize]
        batched_train_out[i] = batched_train[i]
        temp+=1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x) :
    t = np.exp(x)
    return t/ np.sum(t, axis=0)

def relu(x) :
    return np.maximum(x, 0)

def tanh(x) :
    return np.tanh(x)

def d_sigmoid(x) :
    return x * (1-x)

def d_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def d_tanh(x) :
    return (1 - (x ** 2))

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce
def forward_prop(weight, bias, batch_no, no_of_layers, out_dict, gr_truth_out, Z, 
                 hidden_activation, output_activation) :
    for i in range(no_of_layers-1) :
        if(hidden_activation == 'relu') :
            Z[i+1] = np.matmul(weight[i+1],out_dict[i]) + bias[i+1]
            out_dict[i+1] = relu(Z[i+1]) 
        elif(hidden_activation == 'tanh') :
            Z[i+1] = np.matmul(weight[i+1],out_dict[i]) + bias[i+1]
            out_dict[i+1] = tanh(Z[i+1])
        
        else:
            Z[i+1] = np.matmul(weight[i+1],out_dict[i]) + bias[i+1]
            out_dict[i+1] = sigmoid(Z[i+1]) 
    
    if(output_activation == 'sigmoid') :
        Z[no_of_layers] = np.matmul(weight[no_of_layers],out_dict[no_of_layers-1]) + bias[no_of_layers]
        out_dict[no_of_layers] = sigmoid(Z[no_of_layers])
    elif(output_activation == 'relu') :
        Z[no_of_layers] = np.matmul(weight[no_of_layers],out_dict[no_of_layers-1]) + bias[no_of_layers]
        out_dict[no_of_layers] = relu(Z[no_of_layers])
    elif(output_activation == 'linear') :
            Z[no_of_layers] = np.matmul(weight[no_of_layers],out_dict[no_of_layers-1]) + bias[no_of_layers]
            out_dict[no_of_layers] = (Z[no_of_layers])
    elif(output_activation == 'softmax') :
        Z[no_of_layers] = np.matmul(weight[no_of_layers],out_dict[no_of_layers-1]) + bias[no_of_layers]
        out_dict[no_of_layers] = softmax(Z[no_of_layers])

def back_prop(weight, bias, batch_no, no_of_layers, out_dict, gr_truth_out, D_weight,
              D_bias, batchsize, Z, dZ, hidden_activation, output_activation) :
    if output_activation == 'linear':
        dZ[no_of_layers] = out_dict[no_of_layers] - gr_truth_out
        D_weight[no_of_layers] = (1/batchsize) * np.matmul(dZ[no_of_layers], out_dict[no_of_layers-1].T)
        D_bias[no_of_layers] = 1/batchsize * np.sum(dZ[no_of_layers], axis = 1, keepdims = True)
        weight[no_of_layers]-= 0.0001 * D_weight[no_of_layers]
        bias[no_of_layers]-= 0.0001 * D_bias[no_of_layers]
        
    else:
        dZ[no_of_layers] = (out_dict[no_of_layers] - gr_truth_out)*d_relu(out_dict[no_of_layers])
        D_weight[no_of_layers] = (1/batchsize) * np.matmul(dZ[no_of_layers], out_dict[no_of_layers-1].T)
        D_bias[no_of_layers] = 1/batchsize * np.sum(dZ[no_of_layers], axis = 1, keepdims = True)
        weight[no_of_layers]-= 0.0001 * D_weight[no_of_layers]
        bias[no_of_layers]-= 0.0001 * D_bias[no_of_layers]
        
    LR = 0.001
    if(hidden_activation == 'relu') :
        for i in range(no_of_layers-1,0,-1) :
            dZ[i] = np.matmul(weight[i+1].T,dZ[i+1]) * d_relu(Z[i])
            D_weight[i] = 1/batchsize * np.matmul(dZ[i],out_dict[i-1].T)

            D_bias[i] = 1/batchsize * np.sum(dZ[i], axis = 1, keepdims = True)
            weight[i]-= LR * D_weight[i]
            bias[i]-= LR * D_bias[i]
    elif(hidden_activation == 'tanh') :
        for i in range(no_of_layers-1,0,-1) :
            dZ[i] = np.matmul(weight[i+1].T,dZ[i+1]) * d_tanh(Z[i])
            D_weight[i] = 1/batchsize * np.matmul(dZ[i],out_dict[i-1].T)
            D_bias[i] = 1/batchsize * np.sum(dZ[i], axis = 1, keepdims = True)
            weight[i]-= LR * D_weight[i]
            bias[i]-= LR * D_bias[i]
    elif(hidden_activation == 'sigmoid') :
        for i in range(no_of_layers-1,0,-1) :
            dZ[i] = np.matmul(weight[i+1].T,dZ[i+1]) * d_sigmoid(Z[i])
            D_weight[i] = 1/batchsize * np.matmul(dZ[i],out_dict[i-1].T)
            D_bias[i] = 1/batchsize * np.sum(dZ[i], axis = 1, keepdims = True)
            weight[i]-= LR * D_weight[i]
            bias[i]-= LR * D_bias[i]


            