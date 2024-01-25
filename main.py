import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, regularizers
import utils, models, plotting

import sys

print(f"TensorFlow version is {tf.__version__}")

# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Print the names of all available GPUs
    for gpu in gpus:
        print("GPU Name:", gpu.name)
else:
    print("No GPU devices found.")

#load data

dir_path = '../cleaned_resized' #we work with images at resolution 32x64
train_path = dir_path+"/training"
gallery_path = dir_path+"/gallery"
query_path = dir_path+"/query"

x_train, y_train, c_train = utils.load_data( train_path, "training")
x_query, y_query, c_query = utils.load_data( query_path , "query")
x_gallery, y_gallery, c_gallery = utils.load_data( gallery_path, "gallery")

print("\nTraining Data:")
print(f"X shape: {x_train.shape}" )
print(f"Y shape: {y_train.shape}" )
print(f"C shape: {c_train.shape}" )

print("\nQuery Data:")
print(f"X shape: {x_query.shape}" )
print(f"Y shape: {y_query.shape}" )
print(f"C shape: {c_query.shape}" )

print("\nGallery Data:")
print(f"X shape: {x_gallery.shape}" )
print(f"Y shape: {y_gallery.shape}" )
print(f"C shape: {c_gallery.shape}" )

print( f"min value ={np.min( x_train) }")
print( f"max value ={np.max( x_train) }")
            
### Generative Network hyperparameters

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
image_size = (64,32,3)
depths = [48, 96, 192, 384]
block_depth = 2

### Training hyperparameters
batch_size = 16
learning_rate = 0.0001
weight_decay = 1e-4
lambda_factor = .5 #1.0
# cosine_annealing  = 1e-4
num_epochs = 0
epoch_save_weights_frequency = 0

id_embedding_dim = 32
### Create and compile the model

model = models.DiffusionModel(image_size, depths, block_depth, batch_size, id_embedding_dim,max_signal_rate,min_signal_rate,lambda_factor)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error
)

model.normalizer.adapt(np.array(x_train))


### Run training and plot generated images periodically
weights_filename = f"generator_weights.hdf5"
weights_path_model ="weights/"+weights_filename

if os.path.exists(weights_path_model):
    print("load weights from "+weights_path_model )
    model.load_weights( weights_path_model )
else:
    print("no weights found")
    
# Define a Lambda Callback function to save weights
def get_save_weights_callback(weights_path,save_freq=3):
    def save_weights_callback_fun(epoch, logs):
        if epoch>0 and (epoch) % save_freq == 0:
            model.save_weights(weights_path)
        print(f"\tSaved weights")
    return save_weights_callback_fun

swc=None
if epoch_save_weights_frequency>0:
    swc = [callbacks.LambdaCallback( on_epoch_end=get_save_weights_callback( weights_path_model, save_freq=epoch_save_weights_frequency))]

if num_epochs > 0:
    print("starting training")
    x, y = x_train, y_train
    last_train_idx = ( len( x ) // batch_size) * batch_size
    print( f"last train index = {last_train_idx}")
    train_ds = tf.data.Dataset.from_tensor_slices((x[:last_train_idx], y[:last_train_idx] )).batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    model.fit(
        train_ds,
        epochs=num_epochs,
        #batch_size=batch_size
        callbacks=swc
    )
    
    msg=""
    if epoch_save_weights_frequency==0 or num_epochs % epoch_save_weights_frequency!=0:
        msg="saved weights, "
        model.save_weights(weights_path_model)
    
    print(msg+"fit completed.")


while False:
    plotting.plot_conditioning( model, x_train, y_train, no_label=3, no_images=6 , same_noise=True)

while False:
    plotting.plot_random(model,y_train)


######################################################################
# loading and training the auxiliary generator
# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
data_size = 32
#widths = [256,256,256]
widths = [512,512,512]

### Training hyperparameters
batch_size = 256
learning_rate = 0.0001
#weight_decay = 1e-4
num_epochs = 0
epoch_save_weights_frequency = 0

auxiliary_gen = models.DenseDiffusionModel(data_size, widths,max_signal_rate,min_signal_rate)
auxiliary_gen.network.summary()

auxiliary_gen.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error
)

ids_emb = model.label_encoder(np.arange(1,1502))
auxiliary_gen.normalizer.adapt(np.array(ids_emb))

weights_filename = f"auxiliary_generator.hdf5"
weights_path_model ="weights/"+weights_filename

if os.path.exists(weights_path_model):
    print("load weights from "+weights_path_model )
    auxiliary_gen.load_weights( weights_path_model )
else:
    print("no weights found")
    
# Define a Lambda Callback function to save weights
def get_save_weights_callback(weights_path,save_freq=3):
    def save_weights_callback_fun(epoch, logs):
        if epoch>0 and (epoch) % save_freq == 0:
            auxiliary_gen.save_weights(weights_path)
        print(f"\tSaved weights")
    return save_weights_callback_fun

swc=None
if epoch_save_weights_frequency>0:
    swc = [callbacks.LambdaCallback( on_epoch_end=get_save_weights_callback( weights_path_model, save_freq=epoch_save_weights_frequency ))]

if num_epochs > 0:
    auxiliary_gen.fit(
        ids_emb,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=swc
    )
    
    msg=""
    if epoch_save_weights_frequency==0 or num_epochs % epoch_save_weights_frequency!=0:
        msg="saved weights, "
        auxiliary_gen.save_weights(weights_path_model)
    
    print(msg+"fit completed.")

#################################
# inverting the generator

def datagenerator_from_diff(auxiliary_gen,batch_size):
    while True:
        emb_noise = np.random.normal(size=(batch_size,64,32,3))
        emb_ids = auxiliary_gen.generate(batch_size,50)
        generated = model.reverse_diffusion_from_emb(emb_noise, emb_ids, diffusion_steps=10)
        generated = model.denormalize(generated)
        yield generated, emb_ids

#label embedding
Img2Id = models.get_Embedder(image_size,depths,block_depth)

weights_name = 'weights/Img2Id_weights.hdf5' 

optimizer=keras.optimizers.Adam(learning_rate=.00004)
Img2Id.compile(optimizer=optimizer,loss='mse')


batch_size=64
num_epochs = 0

if os.path.exists(weights_name):
    Img2Id.load_weights(weights_name)
    print("loaded weights")
else:
    print("no weights found for Embedder")

if num_epochs > 0:
    print("start training")
    Img2Id.fit(
        from_diff_gen,
        epochs=num_epochs,
        steps_per_epoch = 500
        #batch_size=batch_size
        #callbacks=swc
    )
    Img2Id.save_weights(weights_name)

#assert False

######################################################################
# noise inversion
# not required for person ReID;
# loading too many models could be memory demanding; we do it on demand

if False:
    noise_weights_name = "../weights/Img2noise.hdf5"
    Img2noise = models.get_Unet(image_size,depths,block_depth)

    if os.path.exists(noise_weights_name):
        noise_embedding_model.load_weights(noise_weights_name)
        print("loaded weights")
    else:
        print("no weights found for noise_embedding")

    batch_size = 8
    epochs = 0
    steps_per_epoch = 500
    
    label_encoder = model.label_encoder 
    optimizer=keras.optimizers.Adam(learning_rate=.0001)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        prog_bar = tf.keras.utils.Progbar(steps_per_epoch, unit_name='batch')

        for step in range(steps_per_epoch):

            img_batch, emb_label = next(from_diff_gen)

            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                emb_noise = Img2noise(img_batch)
                generated = model.reverse_diffusion_from_emb(emb_noise, emb_label, diffusion_steps=10)
                generated = model.denormalize(generated)
                loss_value = tf.reduce_mean(tf.math.abs(img_batch - generated))
        
            grads = tape.gradient(loss_value, Img2noise.trainable_weights)
            optimizer.apply_gradients(zip(grads, Img2noise.trainable_weights))
            if step % 5 == 0:
                values = [('loss_img', float(loss_value))]
            
                prog_bar.update(step + 1, values=values)
                
    Img2noise.save_weights(noise_weights_name)
    print("\nsaved weights, training completed")

    plottin.plot_reconstructions(x_gallery[22:30],y_gallery[22:30])

########################################################################

from scipy.spatial.distance import cdist

def compute_euclid_dist_matrix(vv1, vv2):
    return cdist(vv1, vv2, 'euclidean')


#compute distances between all the embeddings between query-gallery pairs
emb_query = Img2Id.predict( x_query )
emb_gallery = Img2Id.predict(x_gallery)

#emb_query = np.load("query_distances.npy")
#emb_gallery = np.load("gallery_distances.npy")

#distances = compute_distance( emb_query, emb_gallery ) 
distances = compute_euclid_dist_matrix( emb_query, emb_gallery ) 

sorted_indices = np.argsort( distances, axis=-1)
print( f"distances shape {distances.shape}" )

#calcolo la rank matrixes di labels ed embeddigns
rm_labels = y_gallery[sorted_indices]
rm_embs = emb_gallery[sorted_indices]
rm_cams = c_gallery[sorted_indices]

print(f"rank_matrix shape {rm_labels.shape}")

########################################################################
# map computation 
np.set_printoptions(threshold=np.inf)

query_labels = y_query
rank_matrix = rm_labels # ordered gallery labels
rank_matrix_cam = rm_cams
query_cams = c_query

lpack = np.tile( np.expand_dims(query_labels,axis=1), (1,rank_matrix.shape[1]))
has_same_id = (rank_matrix == lpack).astype(np.float32)

cpack = np.tile( np.expand_dims(query_cams,axis=1), (1,rank_matrix_cam.shape[1]))

has_same_cam = (rank_matrix_cam == cpack).astype(np.float32)

condition = np.logical_not(np.logical_and( has_same_id == 1, has_same_cam == 1) ) 


def compute_cmc_at_k(is_rel):
    eps= 1e-15
    tp = np.cumsum( is_rel )+eps #true positives
    tr = np.sum( is_rel )+eps #total relevants
    return (tp/tr) #cmc top-20

def compute_ap(is_rel):
    tp = np.cumsum( is_rel )  #true positives
    tr = np.sum( is_rel ) #total relevants
    pos = np.arange( 1, new_row.shape[0]+1) #positions
    precision = tp / pos #precision
    ap = np.sum(precision*is_rel) / tr #average precision
    return ap
    

# due to camera filtering the rank_matrix can be a list of numpys arrays of different lenghts
rm = []  
cmc = [] 
map = [] 

for idx, (row,cond) in enumerate( zip(rank_matrix,condition) ):
    # new_row= row[cond] 
    new_row=row
    rm.append(new_row) 
    
    pid_query= query_labels[idx] 
    is_rel= (new_row == pid_query).astype(np.uint8) #is relevant
    cmc_score= compute_cmc_at_k( is_rel[:25])
    cmc.append( cmc_score )

    ap = compute_ap( is_rel )
    map.append( ap )
    

map_score = np.array( map ).mean(axis=0)
cmc_score = np.array(cmc).mean(axis=0)
print( f"map_score: {map_score}")

print( f"rank-1 acc: {cmc_score[0]}")
print( f"rank-5 acc: {cmc_score[4]}")
print( f"rank-10 acc: {cmc_score[9]}")
print( f"rank-20 acc: {cmc_score[19]}")

def plot(ix):
    y_sample = y_query[ix]
    x_sample = x_query[ix]
    #estrapolo le top-k label
    topk=5
    top_indices = sorted_indices[ix,:topk]
    
    fig, axes = plt.subplots(1,topk+1)
    axes[0].axis("off")
    axes[0].imshow(x_sample)
    axes[0].set_title(f"id {y_sample}")
  
    for i, idx in enumerate(top_indices):
        l = y_gallery[idx]
        img = x_gallery[idx]
        axes[i+1].imshow(img)
        axes[i+1].axis("off")
        axes[i+1].set_title(f"id {l}")

    plt.show()

while True:
    i = np.random.randint(y_query.shape[0])
    plot(i)
    

