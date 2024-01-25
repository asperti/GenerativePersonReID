import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, regularizers
import numpy as np
import math


### Network's Blocks
def sinusoidal_embedding(embedding_dims,c_axis=3):
    def freq(x):
        embedding_min_frequency = 1.0
        embedding_max_frequency = 1000.0
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(embedding_min_frequency),
                tf.math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=c_axis
        )
        return embeddings
    return freq

    
def ResidualBlock(depth):
    def apply(x):
        input_depth = x.shape[3]
        if input_depth == depth:
            residual = x
        else:
            residual = layers.Conv2D(depth, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            depth, kernel_size=3, padding="same",activation='swish')(x)
        x = layers.Conv2D(depth, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(depth, block_depth):
    def apply(x):
        x, id_emb, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        id_emb = layers.Reshape( (1,1, id_emb.shape[-1]) )(id_emb)
        id_emb = layers.UpSampling2D(size=(x.shape[1], x.shape[2]),
                                     interpolation="nearest")(id_emb)
        cat_xemb = layers.Concatenate()([x,id_emb])
        return cat_xemb
    return apply


def UpBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(depth)(x)
        return x

    return apply

def DownBlockNoise(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def DownBlockEncoder(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def LabelUpBlock():
    def apply(x):
        #x = tf.reduce_max(x,axis=2) not good
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.Dense(128, activation="swish")(x)
        x = layers.Dense(32)(x)
        return x
    return apply
        

def get_denoising_network(image_size, depths, block_depth, id_embedding_dim):

    ##label encoder model: embedding ID into a latent space
    label = keras.Input(shape=() )
    cl = layers.Embedding( input_dim=1502, output_dim=id_embedding_dim)( label )
    label_encoder = keras.Model(label, cl)

    ## denoising model: #takes in input a noisy image, a noise rate and a conditioning id embedding

    #input layers of emb encoder
    noisy_images = keras.Input(shape=image_size)
    noise_variances = keras.Input(shape=(1, 1, 1))
    id_embedding = keras.Input(shape=(id_embedding_dim,) )
        
    e = layers.Lambda(sinusoidal_embedding(64,c_axis=3))(noise_variances)
    e = layers.UpSampling2D( size=(image_size[0], image_size[1]), interpolation="nearest")(e)
    
    c = layers.Reshape((1,1,id_embedding_dim))(id_embedding)
    c = layers.UpSampling2D(size=(image_size[0], image_size[1]),interpolation="nearest")(c)
    
    x = layers.Conv2D(depths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e, c])
    
    skips = []
    for depth in depths[:-1]:
        x = DownBlock(depth, block_depth)([x, id_embedding, skips])

    for _ in range(block_depth):
        x = ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]):
        x = UpBlock(depth, block_depth)([x, skips])

    x= layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
    
    emb_encoder = keras.Model([noisy_images, noise_variances, id_embedding], x, name="denoising")

    
    ## main encoder model: #similar to the denoising model, but ID-conditioned
    x  = emb_encoder([ noisy_images, noise_variances, cl] )
    main_model =keras.Model([noisy_images, noise_variances, label], x, name="label_denoising")
    
    return main_model, emb_encoder, label_encoder 


def get_Unet(image_size, depths, block_depth):
    #returns a Unet
    input_images = keras.Input(shape=image_size)

    x = layers.Conv2D(depths[0], kernel_size=1)(input_images)

    skips = []
    for depth in depths[:-1]:
        x = DownBlockNoise(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]): 
        x = UpBlock(depth, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros", name = "output_noise")(x)
    
    return keras.Model(input_images, outputs=x, name="unet")

def get_Embedder(image_size, depths, block_depth):
    #returns a network from images to a flat latent space
    #it is used to invert the generator from images to the latent
    #space for identities
    input_images = keras.Input(shape=image_size)

    x = layers.Conv2D(depths[0], kernel_size=1)(input_images)

    skips = []
    for depth in depths[:-1]:
        x = DownBlockEncoder(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(depths[-1])(x)
    
    l =  LabelUpBlock()(x)
    
    return keras.Model(input_images, outputs=l, name="embedder")


def get_dense_denoising_network(data_size,widths):
    #a dense denoising network, used for the auxiliary generator learning
    #the distribution of id representations into their latent space
    
    noisy_data = keras.Input(shape=data_size)
    noise_variances = keras.Input(shape=1)

    x = noisy_data
    e = layers.Lambda(sinusoidal_embedding(32,c_axis=1))(noise_variances)

    for dim in widths:
        x = layers.Dense(dim,activation='swish')(x)
        x = layers.concatenate([x,e])

    x = layers.Dense(data_size)(x)
    
    main_model =keras.Model([noisy_data, noise_variances], x, name="emb_denoising")
    
    return main_model

##################################
# Classes
##################################

def get_augmentation_pipeline():
    return tf.keras.Sequential([
         layers.RandomFlip(mode="horizontal", seed=42)
    ])

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, batch_size,id_embedding_dim,max_signal_rate,min_signal_rate,lambda_factor):
        super().__init__()
        self.image_size = image_size
        self.augmentation = get_augmentation_pipeline()
        
        self.normalizer = layers.Normalization()
        self.network, self.network_emb, self.label_encoder = get_denoising_network(image_size, widths, block_depth, id_embedding_dim)
        self.batch_size = batch_size
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        self.lambda_factor = lambda_factor

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.regularization_loss_tracker = keras.metrics.Mean(name="c_loss")

    def load_weights(self, filename):
        self.network.load_weights(filename)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, 
                self.image_loss_tracker,
                self.regularization_loss_tracker
                ]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, label, training):
        #network = self.network
        # predict noise component and calculate the image component using it
        
        cl = self.label_encoder(label)
        
        pred_noises = self.network_emb([noisy_images, noise_rates**2, cl], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, label, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        if num_images is None:
            num_images = 10
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
       
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise( noisy_images, 
                                                    noise_rates, signal_rates, label, training=False )

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images
    
    def denoise_from_emb(self, noisy_images, noise_rates, signal_rates, emb_label, training):
        network = self.network_emb
        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2, emb_label], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion_from_emb(self, initial_noise, emb_label, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        if num_images is None:
            num_images = 10
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
       
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise_from_emb( noisy_images, noise_rates, signal_rates, emb_label, training=False )

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images
    
    def generate(self, num_images, label, diffusion_steps, same_noise=False):
        #num_images => immagini da creare data 1 sola label
        #label => condizionamento di identità
        
        
        # print( f"labels shape {label.shape}")
        no_labels = label.shape[0]
        # print(f"label prima il repeat {label}")
        input_label = np.repeat( label, num_images )
        # print(f"label dopo il repeat {input_label}")
        
        if same_noise:
            noise_shape = (num_images,) + self.image_size
            initial_noise = tf.random.normal(shape=noise_shape)
            initial_noise = np.tile( initial_noise, (no_labels, 1,1,1) )
           
        else:
            noise_shape = (num_images*no_labels,) + self.image_size
            initial_noise = tf.random.normal(shape=noyise_shape)
        
        # print( f"input_noise shape { initial_noise.shape }" )
        # print( f"noise_shape { noise_shape }" )
       
        generated_images = self.reverse_diffusion(initial_noise, input_label, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images
    
    def train_step(self, input_data):
        images, label = input_data
        
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        images = self.augmentation( images, training=True )
        
        noises_shape = (self.batch_size,) + self.image_size 
        noises = tf.random.normal(shape=noises_shape )
        
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their componenainable_weights)
            emb_labels = self.label_encoder(label)
            pred_noises = self.network_emb([noisy_images, noise_rates**2, emb_labels], training=True)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            
            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
        
            regularization_loss = self.lambda_factor*(tf.reduce_mean(emb_labels**2,axis=1))
            training_loss = tf.reduce_sum(noise_loss, axis=(1,2)) + regularization_loss

        gradients = tape.gradient(training_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)
        return {m.name: m.result() for m in self.metrics }


########################################

#next class is for the auxiliary generator
#we found more convenient to predict the signal instead of the noise

class DenseDiffusionModel(keras.Model):
    def __init__(self, data_size,widths,max_signal_rate,min_signal_rate):
        super().__init__()
        self.data_size = data_size
        self.normalizer = layers.Normalization()
        self.network = get_dense_denoising_network(data_size, widths)
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.data_loss_tracker = keras.metrics.Mean(name="i_loss")

    def load_weights(self, filename):
        self.network.load_weights(filename)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, 
                self.data_loss_tracker,
                ]

    def denormalize(self, data):
        # convert data back to original range
        data = self.normalizer.mean + data * self.normalizer.variance**0.5
        return data

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        #print(f"diffusion angles: {diffusion_angles}")
        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_data, noise_rates, signal_rates, training):
        # predict noise component and calculate the image component using it
        
        pred_data = self.network([noisy_data, noise_rates**2], training=training)
        pred_noises = (noisy_data - signal_rates * pred_data) / noise_rates

        return pred_noises, pred_data

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        data_size = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_data = initial_noise
       
        for step in range(diffusion_steps):
            noisy_data = next_noisy_data

            # separate the current noisy image to its components
            diffusion_times = tf.ones((data_size,1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            #print("noise_rates: ",tf.shape(noise_rates))
            pred_noises, pred_data = self.denoise(noisy_data, noise_rates, signal_rates, training=False)

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_data = (
                next_signal_rates * pred_data + next_noise_rates * pred_noises
            )
            # new noisy data will be used in the next step

        return pred_data
    
    def generate(self, num_data, diffusion_steps, same_noise=False):
        #num_data: for each noise
        
        if same_noise:
            noise_shape = (1,self.data_size)
            initial_noise = tf.random.normal(shape=noise_shape )
            initial_noise = np.tile( initial_noise, (num_data, 1) )
           
        else:
            noise_shape = (num_data,self.data_size)
            initial_noise = tf.random.normal(shape=noise_shape)

        generated_data = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_data)
        return generated_images
    
    def train_step(self, input_data):
        #version reconstructing the signal
        data = input_data
        print("data: ",tf.shape(data))
        
        # normalize images to have standard deviation of 1, like the noises
        data = self.normalizer(data, training=True)
        
        noises_shape = (tf.shape(data)[0],self.data_size)
        noises = tf.random.normal(shape=noises_shape)
        
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(tf.shape(data)[0],1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        print(tf.shape(noise_rates),tf.shape(signal_rates))
        # mix the images with noises accordingly
        noisy_data = signal_rates * data + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their componenainable_weights)
        
            pred_data = self.network([noisy_data, noise_rates**2], training=True)
            pred_noises = (noisy_data - signal_rates * pred_data) / noise_rates
            
            noise_loss = self.loss(noises, pred_noises)  # used as metric
            data_loss = self.loss(data, pred_data)  # used for training
        
            training_loss = data_loss 

        gradients = tape.gradient(training_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.data_loss_tracker.update_state(data_loss)

        print("DONE")
        return {m.name: m.result() for m in self.metrics }
                                                 








