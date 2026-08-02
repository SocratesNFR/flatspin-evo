import torch
from torch import nn, optim
import lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms
from lightning.pytorch.callbacks import EarlyStopping


# Define the Encoder (2 input channels)
class Encoder(nn.Module):
    def __init__(self, latent_size=6, features=128):
        super().__init__()
        self.conv1 = nn.Conv2d(2, features, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(features, features, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(features, features, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * features, 2048)
        self.fc_mean = nn.Linear(2048, latent_size)
        self.fc_logvar = nn.Linear(2048, latent_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


# Define the Decoder (2 output channels)
class Decoder(nn.Module):
    def __init__(self, latent_size=6, features=128):
        super().__init__()
        self.fc = nn.Linear(latent_size, 2048)
        self.fc2 = nn.Linear(2048, 4 * 4 * features)
        self.deconv1 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(features, 2, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        z = torch.relu(self.fc(z))
        z = torch.relu(self.fc2(z))
        z = z.view(-1, 128, 4, 4)
        z = torch.relu(self.deconv1(z))
        z = torch.relu(self.deconv2(z))
        z = self.deconv3(z)  # Output is continuous (no sigmoid)
        return z

def edge_loss(x, x_hat):
    x_edges = TF.gaussian_blur(x, kernel_size=3) - x
    x_hat_edges = TF.gaussian_blur(x_hat, kernel_size=3) - x_hat

    return F.mse_loss(x_hat_edges, x_edges)

# Define the VAE
class LitVAE(L.LightningModule):
    def __init__(self, encoder, decoder, beta=1.0, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.lr =lr
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std



    def training_step(self, batch, batch_idx):
        x, _ = batch  # Load input image
        mean, logvar = self.encoder(x)  # Encode
        z = self.reparameterize(mean, logvar)  # Sample latent vector
        x_hat = self.decoder(z)  # Decode back

        # Compute MSE reconstruction loss
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

        
        # mse_loss = F.mse_loss(x_hat, x)
        # edge_loss_val = edge_loss(x, x_hat)
        # recon_loss = mse_loss + 0 * edge_loss_val

        # ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
        # recon_loss = 1 - ssim(x_hat, x)


        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total β-VAE loss
        loss = recon_loss + self.beta * kl_loss

        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("kl_loss", kl_loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # Encoder function to get latent representations
    def encode(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z

    # Decoder function to get reconstructed images
    def decode(self, z):
        return self.decoder(z)


# Custom Dataset for 2-channel images
class CustomDataset(Dataset):
    def __init__(self, np_array_list, transform=None):
        self.data = np_array_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)  # Convert to float
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label


def encode_images(vae, np_images):
    transform = transforms.ToTensor()  # Ensure it's in the right format
    vae.eval()
    with torch.no_grad():
        # Convert list of np arrays to a tensor and ensure dtype is float32
        images = torch.stack([transform(img).float() for img in np_images])  # Convert list of np arrays
        mean, _ = vae.encoder(images)  # Get the mean encoding
    return mean.numpy()  # Convert back to NumPy if needed

def encode_decode_images(vae, np_images):
    # Step 1: Transform the images from NumPy arrays to torch tensors
    transform = transforms.ToTensor()  # Converts NumPy arrays to torch tensor
    vae.eval()  # Set the model to evaluation mode
    
    # Prepare the images for batch processing
    images = torch.stack([transform(img).float() for img in np_images])  # Convert list of np arrays to tensor batch
    
    # Step 2: Pass the images through the encoder
    with torch.no_grad():
        mean, _ = vae.encoder(images)  # Get the mean encoding (latent space representation)
        
        # Step 3: Decode the latent representations to get the reconstructed images
        reconstructed_images = vae.decoder(mean)  # Decode the mean (using the decoder)
        
        # Step 4: Convert the reconstructed images back to NumPy arrays if needed
        reconstructed_images_np = reconstructed_images.cpu().numpy()  # Convert to NumPy array (move to CPU if on GPU)
        reconstructed_images_np = reconstructed_images_np.transpose(0, 2, 3, 1)  # Change back to (batch_size, height, width, channels)

    return reconstructed_images_np


# Function to train the VAE
def train_vae(np_array_list, vae=None, beta=4.0, latent_size=6, max_epochs=10, batch_size=24, lr=1e-3, patience=3):
    """
    Train a VAE on the given list of 2-channel images.

    Args:
        np_array_list (list): List of NumPy arrays of shape (H, W, 2).
        vae (LitVAE, optional): Pre-existing VAE model. If None, a new one is created.
        beta (float): Weight for the KL divergence in the loss function.
        latent_size (int): Size of the latent space.
        max_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Trained VAE model.
    """

    # Convert NumPy images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor (keeps values in [0,1])
    ])

    # Create dataset and dataloader
    dataset = CustomDataset(np_array_list, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # If no VAE is provided, create a new one
    if vae is None:
        encoder = Encoder(latent_size=latent_size)
        decoder = Decoder(latent_size=latent_size)
        vae = LitVAE(encoder, decoder, beta=beta, lr=lr)

    # Initialize the trainer
    trainer = L.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor="train_loss", patience=patience)])

    # Train the model
    trainer.fit(vae, train_loader)

    return vae

if __name__ == "__main__":
    # Example: Training with a new dataset
    np_array_list = [np.random.rand(32, 32, 2).astype(np.float32) for _ in range(1000)]
    vae = train_vae(np_array_list, max_epochs=10, beta=4.0)

    # Example: Training with an existing VAE on new data
    new_data = [np.random.rand(32, 32, 2).astype(np.float32) for _ in range(500)]
    vae = train_vae(new_data, vae=vae, max_epochs=5)  # Fine-tuning on new data
