
inputs:
 - N_INPUTS = 28*28 = 784 float 0..1.0 f64

hidden layers:
 - N_LAYERS = 2
 - N_L1 = 28*28/4 = 196
 - N_L2 = 28*28/4/4 = 49

outputs:
 - N_OUTPUTS = 10 float 0..1.0 f64

training data:
 - 60000 sample inputs
 - 60000 labels

training data chunk:
 - BATCH_SIZE=100 inputs + labels


sigmoid = e^x / (e^x + 1)


memory:
inputs       = matrix<BATCH_SIZE, N_INPUTS>

l1_weights   = matrix<N_INPUTS, N_L1>
l1_activations = matrix<BATCH_SIZE, N_L1>

l2_weights     = matrix<N_L1, N_L2>
l2_activations = matrix<BATCH_SIZE, N_L2>

outputs_weights = matrix<N_L2, N_OUTPUTS>

true_outputs = matrix<BATCH_SIZE, N_OUTPUTS>
outputs      = matrix<BATCH_SIZE, N_OUTPUTS>

error = vector<N_OUTPUTS>
cost  = vector<N_OUTPUTS>

regularization = f64


code:
init_with_training_data(inputs, true_outputs, disk)
init_random_weights(l1_weights, l2_weights, seed)

# forward run
l1_activations = sigmoid(inputs*l1_weights)
l2_activations = sigmoid(l1_activations*l2_weights)
outputs = sigmoid(l2_activations*outputs_weights)

# error
error = per_col_sum(square(outputs - true_outputs))/BATCH_SIZE
regularization = sum_square_each(l1_weights, l2_weights, outputs_weights) 
cost = error + reg_coeff*regularization

# backward run
outputs_weight_correction = 2*l2_activations.T*(outputs - true_outputs)*outputs
#TODO: check this makes sense!









