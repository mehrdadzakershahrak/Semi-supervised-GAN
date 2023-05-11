# Required Libraries
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, LeakyReLU, Dropout, Flatten, Lambda, Activation, Reshape, Conv2DTranspose
from keras.utils.vis_utils import plot_model
from keras.datasets.mnist import load_data
from keras import backend
from tensorflow.keras.optimizers import Adam
from numpy import expand_dims, zeros, ones, asarray, randn, randint
from matplotlib import pyplot

# Defining the discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
    # Image input
    in_image = Input(shape=in_shape)
    
    # Downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Flatten feature maps
    fe = Flatten()(fe)
    
    # Dropout
    fe = Dropout(0.4)(fe)
    
    # Output layer nodes
    fe = Dense(n_classes)(fe)
    
    # Supervised Output
    c_out_layer = Activation('softmax')(fe)
    # Define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    
    # Unsupervised Output
    d_out_layer = Lambda(custom_activation)(fe)
    # Define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    return d_model, c_model

# Defining the generator model
def define_generator(latent_dim):
    # Image generator input
    in_lat = Input(shape=(latent_dim,))
    
    # Foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    
    # Upsample to 14x14
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    # Upsample to 28x28
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    # Output layer
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    
    # Define model
    model = Model(in_lat, out_layer)
    
    return model

# Defining the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # Make weights in the discriminator not trainable
    d_model.trainable = False
    # Connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # Define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load real samples
def load_real_samples():
    # Load dataset
    (trainX, trainy), (_, _) = load_data()
    # Expand to 3D, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # Convert from ints to floats
    X = X.astype('float32')
    # Scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]

# Select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=100, n_classes=10):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        # Get all images for this class
        X_with_class = X[y == i]
        # Choose random instances
        ix = randint(0, len(X_with_class), n_per_class)
        # Add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)

# Select real samples
def generate_real_samples(dataset, n_samples):
    # Split into images and labels
    images, labels = dataset
    # Choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # Select images and labels
    X, labels = images[ix], labels[ix]
    # Generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # Generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # Reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# Use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # Generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs
    images = generator.predict(z_input)
    # Create class labels
    y = zeros((n_samples, 1))
    return images, y

# Generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
    # Prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # Scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # Plot images
    for i in range(100):
        # Define subplot
        pyplot.subplot(10, 10, 1 + i)
        # Turn off axis
        pyplot.axis('off')
        # Plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # Save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # Evaluate the classifier model
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
        print('Classifier Accuracy: %.3f%%' % (acc * 100))
    # Save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # Save the classifier model
    filename3 = 'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# Train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=100):
    # Select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset)
    print(X_sup.shape, y_sup.shape)
    # Calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # Calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # Calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # Manually enumerate epochs
    for i in range(n_steps):
        # Update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # Update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # Update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # Summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        # Evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, dataset)

# Size of the latent space
latent_dim = 100
# Create the discriminator models
d_model, c_model = define_discriminator()
# Create the generator
g_model = define_generator(latent_dim)
# Create the gan
gan_model = define_gan(g_model, d_model)
# Load image data
dataset = load_real_samples()
# Train model
train(g_model, d_model, c_model, gan_model, dataset, latent_dim)


