import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from matplotlib import pyplot as plt
import pandas as pd
import random

#%% select a portion of data from database
from gemmi import cif
number = []
I = []
for i in range(0, 1000000):#1667834
    try: 
        doc = cif.read_file(".../MP_cifs_processed/mp-"+str(i)+".cif")
        block = doc.sole_block()       
        symmetry_Int_Tables_number = block.find_value('_symmetry_Int_Tables_number')
        symmetry_Int_Tables_number = int(symmetry_Int_Tables_number)
        number.append(symmetry_Int_Tables_number)
        I.append(i)
    except:
        pass
    
    if i % 100000 == 0:
        print(i)

number = np.array(number)
labels = np.empty(len(number))
labels[:] = np.NaN
# 7 crystal systems
labels[(number >= 1) & (number <= 2)] = 0
labels[(number >= 3) & (number <= 15)] = 1
labels[(number >= 16) & (number <= 74)] = 2
labels[(number >= 75) & (number <= 142)] = 3
labels[(number >= 143) & (number <= 167)] = 4
labels[(number >= 168) & (number <= 194)] = 5
labels[(number >= 195) & (number <= 230)] = 6

for label_no in range(7):
    print(np.sum(labels == label_no))

I = np.array(I)
i0 = I[labels == 0]
i1 = I[labels == 1]
i2 = I[labels == 2]
i3 = I[labels == 3]
i4 = I[labels == 4]
i5 = I[labels == 5]
i6 = I[labels == 6]
train_crystal_size = 500
total_crystal_size = 1000
file_id = np.concatenate((i0[:train_crystal_size], i1[:train_crystal_size], i2[:train_crystal_size], i3[:train_crystal_size], i4[:train_crystal_size], i5[:train_crystal_size], i6[:train_crystal_size],
                          i0[train_crystal_size:total_crystal_size], i1[train_crystal_size:total_crystal_size], i2[train_crystal_size:total_crystal_size], i3[train_crystal_size:total_crystal_size], i4[train_crystal_size:total_crystal_size], i5[train_crystal_size:total_crystal_size], i6[train_crystal_size:total_crystal_size]))


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


beam_direction_round = [[0,0,1]]
directions = []
for i in range(6):
    for j in range(6):
        for k in range(6):
            # print([i,j,k])
            angles = []
            for bd in beam_direction_round:
                angle = angle_between(bd, [i,j,k])
                angles.append(angle)
            if np.nanmin(np.asarray(angles)) > 10/180*np.pi:
                beam_direction_round.append([i,j,k])

beam_direction_round = np.asarray(beam_direction_round)
beam_direction_round.shape

#%% load vector maps
points1 = []
points1_1 = []
points1_2 = []
beam_direction_size = len(beam_direction_round)
count = 0
for i_image in file_id:
    count = count + 1
    # print(count)
    for bd in beam_direction_round[:beam_direction_size]:
        data = pd.read_csv('.../ED_simulated_' +
                           str(bd[0]) + '_' + str(bd[1]) + '_' + str(bd[2]) + 
                           '/mp-'+str(i_image)+'ED_conv.csv')
        
        position = data['Position']
        position = position.tolist()
        position_list = []
        for i in range(len(position)):
            position[i] = str(position[i]).replace('[' , '' )
            position[i] = str(position[i]).replace(']' , '' )
            a = []
            for num_str in position[i].split():
                num_float = float(num_str)
                a.append(num_float)
            position_list.append(a)
        position = np.asarray(position_list)
        
        x = position[:,0]
        y = position[:,1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        intensity = data['Intensity (norm)']
        intensity_list = intensity.tolist()
        intensity = np.asarray(intensity)
        point_array_polar = np.column_stack((rho, phi, intensity))
        point_list = point_array_polar.tolist()       
        point_list = sorted(point_list, key=lambda x: (x[1], x[0], x[2])) # sort y-x-intensity
        point_list_1 = sorted(point_list, key=lambda x: (x[0], x[1], x[2])) # sort x-y-intensity
        point_list_2 = sorted(point_list, key=lambda x: (x[2], x[1], x[0])) # sort intensity-y-x
        points1.append(point_list)
        points1_1.append(point_list_1)
        points1_2.append(point_list_2)
        
    if count % 1000 == 0:
        print(count)

for sublist in points1:
    sublist[:] = sublist + [[0,0,0]] * (800 - len(sublist)) # 483

for sublist in points1_1:
    sublist[:] = sublist + [[0,0,0]] * (800 - len(sublist))

for sublist in points1_2:
    sublist[:] = sublist + [[0,0,0]] * (800 - len(sublist))

points1 = np.asarray(points1)
points1_1 = np.asarray(points1_1)
points1_2 = np.asarray(points1_2)
points = np.concatenate((points1, points1_1, points1_2), axis=2)

#%% Load labels
number = []
I = []
count = 0
for i in file_id:
    count = count + 1
    for bd in beam_direction_round[:beam_direction_size]:
        doc = cif.read_file(".../MP_cifs_processed/mp-"+str(i)+".cif")
        block = doc.sole_block()       
        symmetry_Int_Tables_number = block.find_value('_symmetry_Int_Tables_number')
        symmetry_Int_Tables_number = int(symmetry_Int_Tables_number)
        number.append(symmetry_Int_Tables_number)
        I.append(i)
    
    if count % 1000 == 0:
        print(count)


number = np.array(number)
labels = np.empty(len(number))
labels[:] = np.NaN
# 7 crystal systems
labels[(number >= 1) & (number <= 2)] = 0
labels[(number >= 3) & (number <= 15)] = 1
labels[(number >= 16) & (number <= 74)] = 2
labels[(number >= 75) & (number <= 142)] = 3
labels[(number >= 143) & (number <= 167)] = 4
labels[(number >= 168) & (number <= 194)] = 5
labels[(number >= 195) & (number <= 230)] = 6

for label_no in range(7):
    print(np.sum(labels == label_no))


training_size = train_crystal_size*7*beam_direction_size
shuffle_id = list(range(training_size))
random.Random(4).shuffle(shuffle_id)


train_points_1 = points1[shuffle_id,:,:]
train_points_2 = points1_1[shuffle_id,:,:]
train_points_3 = points1_2[shuffle_id,:,:]
train_labels = labels[shuffle_id]

test_points_1 = points1[training_size:,:,:]
test_points_2 = points1_1[training_size:,:,:]
test_points_3 = points1_2[training_size:,:,:]
test_labels = labels[training_size:]

validation_points_1 = test_points_1
validation_points_2 = test_points_2
validation_points_3 = test_points_3
validation_labels = test_labels

#%% Prepare the dataset.
batch_size = 64
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
validation_labels = tf.keras.utils.to_categorical(validation_labels)

x_train_1 = train_points_1
x_train_2 = train_points_2
x_train_3 = train_points_3
y_train = train_labels

x_val_1 = validation_points_1
x_val_2 = validation_points_2
x_val_3 = validation_points_3
y_val = validation_labels 

x_test_1 = test_points_1
x_test_2 = test_points_2
x_test_3 = test_points_3
y_test = test_labels

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_1, x_train_2, x_train_3, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val_1, x_val_2, x_val_3, y_val))
val_dataset = val_dataset.batch(batch_size)

#%% Build a model
np.random.seed(1234)
tf.random.set_seed(1234)
val_accuray = []

input_layer_1 = tf.keras.layers.Input(shape=(points1.shape[1], points1.shape[2]))
conv_layer1_1 = tf.keras.layers.Conv1D(32, 5, activation='relu')(input_layer_1)
conv_layer2_1 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer1_1)
conv_layer3_1 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer2_1)
conv_layer4_1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(conv_layer3_1)
conv_layer5_1 = tf.keras.layers.Conv1D(256, 3, activation='relu')(conv_layer4_1)
global_pooling_layer_1 = tf.keras.layers.GlobalMaxPooling1D()(conv_layer5_1)

input_layer_2 = tf.keras.layers.Input(shape=(points1_1.shape[1], points1_1.shape[2]))
conv_layer1_2 = tf.keras.layers.Conv1D(32, 5, activation='relu')(input_layer_2)
conv_layer2_2 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer1_2)
conv_layer3_2 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer2_2)
conv_layer4_2 = tf.keras.layers.Conv1D(64, 3, activation='relu')(conv_layer3_2)
conv_layer5_2 = tf.keras.layers.Conv1D(256, 3, activation='relu')(conv_layer4_2)
global_pooling_layer_2 = tf.keras.layers.GlobalMaxPooling1D()(conv_layer5_2)

input_layer_3 = tf.keras.layers.Input(shape=(points1_2.shape[1], points1_2.shape[2]))
conv_layer1_3 = tf.keras.layers.Conv1D(32, 5, activation='relu')(input_layer_3)
conv_layer2_3 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer1_3)
conv_layer3_3 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_layer2_3)
conv_layer4_3 = tf.keras.layers.Conv1D(64, 3, activation='relu')(conv_layer3_3)
conv_layer5_3 = tf.keras.layers.Conv1D(256, 3, activation='relu')(conv_layer4_3)
global_pooling_layer_3 = tf.keras.layers.GlobalMaxPooling1D()(conv_layer5_3)

dense_layer1 = tf.keras.layers.Dense(256, activation='relu')(tf.keras.layers.Concatenate()([global_pooling_layer_1, global_pooling_layer_2, global_pooling_layer_3]))
dense_layer2 = tf.keras.layers.Dense(128, activation='relu')(dense_layer1)
output_layer = tf.keras.layers.Dense(7)(dense_layer2)
model = tf.keras.models.Model(inputs=[input_layer_1, input_layer_2, input_layer_3], outputs=output_layer)
model.summary()

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# convert logits to evidence
def relu_evidence(logits):
    return tf.nn.relu(logits)

# loss function
K = 7 # number of classes

def KL(alpha):
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.math.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.math.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.math.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.math.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)
    kl = tf.math.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, logits, epoch):
    logits2evidence=relu_evidence    
    evidence = logits2evidence(logits)
    alpha = evidence + 1
    S = tf.math.reduce_sum(alpha, axis=1, keepdims=True) 
    E = alpha - 1
    m = alpha / S
    A = tf.math.reduce_sum((p-m)**2, axis=1, keepdims=True) 
    B = tf.math.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
    annealing_step = 50
    annealing_coef = tf.math.minimum(1.0,tf.cast(epoch/annealing_step,tf.float32))        
    alp = E*(1-p) + 1 # equivalent to alp = alpha*(1-p) + p 
    C =  annealing_coef * KL(alp)
    return tf.math.reduce_mean((A + B) + C)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

def logits2probability(logits):
    logits2evidence=relu_evidence 
    evidence = logits2evidence(logits)
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    prob = alpha / S
    return prob

#%% Training
@tf.function
def train_step(x1, x2, x3, y, epoch):
    with tf.GradientTape() as tape:
        logits = model([x1, x2, x3], training=True)
        loss_value = mse_loss(y, logits, epoch)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    prob = logits2probability(logits)
    train_acc_metric.update_state(y, prob)
    return loss_value

@tf.function
def test_step(x1, x2, x3, y):
    val_logits = model([x1, x2, x3], training=False)
    val_prob = logits2probability(val_logits)
    val_acc_metric.update_state(y, val_prob)

epochs = 10
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    
    # Iterate over the batches of the dataset.
    for step, (x_batch_train_1, x_batch_train_2, x_batch_train_3, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train_1, x_batch_train_2, x_batch_train_3, y_batch_train, epoch)
        
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    
    # Run a validation loop at the end of each epoch.
    for x_batch_val_1, x_batch_val_2, x_batch_val_3, y_batch_val in val_dataset:
        test_step(x_batch_val_1, x_batch_val_2, x_batch_val_3, y_batch_val)
    
    val_acc = val_acc_metric.result() 
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    
model.save_weights('...'+'.h5')

#%% Prediction
logits_pred = model.predict([x_test_1, x_test_2, x_test_3])
logits2evidence=relu_evidence 
evidence = logits2evidence(logits_pred)
alpha = evidence + 1
S = tf.reduce_sum(alpha, axis=1, keepdims=True)
u = K / S #uncertainty
prob = alpha / S
b = evidence / S
pred_labels = prob.numpy().argmax(axis=-1)
true_labels = y_test.argmax(axis=-1)
misclasification_id = np.where(pred_labels != true_labels)
correct_clasification_id = np.where(pred_labels == true_labels)

from sklearn.metrics import accuracy_score
accuracy_score(true_labels, pred_labels)


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, pred_labels)
class_names = ['triclinic',  'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)# np.arange(32)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format = '') #, ax=ax
plt.show()