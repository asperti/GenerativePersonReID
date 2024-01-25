#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

##########################
#plot a list of images with specified id

def simple_plot(images,indexes):
    n = len(indexes)
    fig, axes = plt.subplots(1,n)
    for i in range(n):
        axes[i].axis("off")
        axes[i].imshow(images[i])
        axes[i].set_title(f"id {indexes[i]}")
    plt.show()

###########################
#plot randomly generated images:
# model: the generative model
# labels: the conditional id
# no_images: number of images to be generated for each id
# diffusion_steps: number of steps off the denoising process

def plot_random(model, labels, no_images=6, diffusion_steps=10):
    ids = np.unique(labels)
    print(ids.shape)
    ids_emb = model.label_encoder(ids)
    ids_emb_mean = np.mean(ids_emb,axis=0)
    ids_emb_std = np.std(ids_emb,axis=0)

    emb_label = np.random.normal(ids_emb_mean, ids_emb_std,size=(no_images,32))
    emb_noise = np.random.normal(size=(no_images,64,32,3))
    generated = model.reverse_diffusion_from_emb(emb_noise, emb_label, diffusion_steps=10)
    generated = model.denormalize(generated)
    
    fig, axes = plt.subplots(1, no_images)
                                 
    for col in range(no_images):
        ax = axes[col]
        ax.imshow( generated[col] )
        #ax.set_title(f"Gen")
        ax.axis("off")

    plt.tight_layout()    
    plt.show()

###################################
#Show the effect of conditional generation with different ids
#The first image in the row is a sample of the given id, the following
#images are generated. The noise in each column can be always the same,
#or can be different for each generattion (default is same_noise=True)

def plot_conditioning( model, images, labels, no_label=3, no_images=6 , diffusion_steps=10, same_noise=True):
    #images and labels are x and y set (train or gallery)
    #no_lables is the number of different identities

    # rows, cols = no_label,no_images
    random_indexes = np.random.choice(labels, no_label, replace=False)

    random_labels = labels[random_indexes]
    retrieved_images = images[random_indexes]

    generated_images = model.generate(
        num_images=no_images,
        label=random_labels,
        diffusion_steps=diffusion_steps,
        same_noise=same_noise
    )
    
    idxs_len = len(generated_images)
    fig, axes = plt.subplots(no_label, no_images+1 )
    if no_label==1:
        ax = axes[0]
        ax.imshow( retrieved_images[0] )
        ax.set_title(f"Id {random_labels[0] }")
        ax.axis("off")
        
        for col in range(no_images):
            ax = axes[col+1]
            ax.imshow( generated_images[col] )
            ax.set_title(f"Gen")
            ax.axis("off")
    
    else: #no_labels>1
        for row in range(no_label):
            ax = axes[row,0]
            ax.imshow( retrieved_images[row] )
            ax.set_title(f"Id {random_labels[row] }")
            ax.axis("off")
    
        for row in range(no_label):
            for col in range(no_images):
                index = row*no_images + col
                # print( f"immagine generata num [{index}]")
                ax = axes[row,col+1]
                ax.imshow( generated_images[index] )
                ax.set_title(f"Noise {col}")
                ax.axis("off")

    # Mostra il grafico
    plt.tight_layout()    
    plt.show()

##############################
def plot_reconstructions(lable_encoder,noise_encoder,generator,images, labels):
    emb_label =label_encoder.predict(labels)  # extract identity
    emb_noise = noise_encoder.predict(images) # extract noise
    generated = generator.reverse_diffusion_from_emb(emb_noise, emb_label, diffusion_steps=10) # regenerate the input image
    generated = model.denormalize(generated)

    renorm_noise = np.clip(emb_noise, -2.5, 2.5)
    renorm_noise = (renorm_noise + 2.5) / 5.

    fig,axes = plt.subplots(3, generated.shape[0])

    for i, (orig,gen) in enumerate(zip(images, generated)):
        axes[0,i].imshow(orig);
        axes[0,i].set_title(f"Id {labels[i]}")
        axes[0,i].axis("off")

        axes[1,i].imshow(renorm_noise[i])
        axes[1,i].set_title(f"Noise")
        axes[1,i].axis("off")
        
        axes[2,i].imshow(gen);
        axes[2,i].set_title("ReGen")
        axes[2,i].axis("off")

    plt.tight_layout()    
    plt.show()
