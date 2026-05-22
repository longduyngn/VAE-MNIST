import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



#define class for sampling
#which we use to take sample from laten space
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        #get shape
        batch, dim = z_mean.shape
        #get epsilon from Normal
        # reparameterization trick
        epsilon = Normal(0,1).sample((batch,dim)).to(z_mean.device)

        #apply reparameterization trick
        return z_mean + epsilon * torch.exp(0.5 * z_log_var)


#encoder : conv + stride to build up feature, then flatten and 1 layer of FCL to produce mu, log_var.
class Encoder(nn.Module):
    def __init__(self, image_size, embedding_dim):
        super(Encoder, self).__init__()

        #conv layer for downsampling & feature extraction
        self.conv1 = nn.Conv2d(1, 32,3,stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        #def the flattening layer
        self.flatten = nn.Flatten()

        #def the Linear layer to output desired embed dim
        self.fc_mean = nn.Linear(
            128 * (image_size // 8) * (image_size // 8), embedding_dim
        )
        self.fc_log_var = nn.Linear(
            128 * (image_size // 8) * (image_size // 8), embedding_dim
        )
        # initialize the sampling layer
        self.sampling = Sampling()

    def forward(self, x):
        # conv
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #flatten: (batch_size, *shape_before_flatten) -> (batch_size, flattened size)
        x = self.flatten(x)
        # get mu, log_var of latent space distribution: (batch_size, flattened size) ->(batch_size,embedding_dim)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        # sample a latent vector using the reparameterization trick
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


#decoder
class Decoder(nn.Module):
    def __init__(self,embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()

        #define FCL layers to convert back to shape before flatten:
        self.fc = nn.Linear(
            embedding_dim,
            shape_before_flattening[0] *
            shape_before_flattening[1] *
            shape_before_flattening[2]
        )

        #reshape back to original
        self.reshape = lambda x: x.view(-1, *shape_before_flattening)

        # conv to upsamples
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self,x):
        # latent - FC : (batch_size, embedding_dim) -> (batch_size, flatten)
        x = self.fc(x)
        # reshape: (batch_size, flatten) -> (batch_size, shape_before_flatten)
        x = self.reshape(x)
        # apply conv with relu
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))

        #final conv with sigmoid to produce output: (batch_size, shape of the image)
        x = torch.sigmoid(self.deconv3(x))
        return x


# VAE
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # pass the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x)

        # reconstruct image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction
