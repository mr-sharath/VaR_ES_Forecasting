# scripts/train_es.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Configuration
VAR_PATH = 'results/var_predictions/BLK.csv'
SAVE_PATH = 'models/es_gan.h5'
LATENT_DIM = 16
EPOCHS = 3000
BATCH_SIZE = 32

def load_and_preprocess():
    data = pd.read_csv(VAR_PATH, index_col=0, parse_dates=True)
    
    # Ensure we have returns and VaR predictions
    assert 'return' in data.columns and 'VaR_95' in data.columns
    
    # Get tail scenarios
    tail_mask = data['return'] < data['VaR_95']
    tail_data = data[tail_mask][['return', 'VaR_95']].values
    
    # Normalize
    scaler = MinMaxScaler()
    return scaler.fit_transform(tail_data), scaler

def build_generator():
    inputs = tf.keras.Input(shape=(LATENT_DIM + 1,))  # noise + conditional VaR
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)  # Generate return
    return tf.keras.Model(inputs, output)

def build_discriminator():
    inputs = tf.keras.Input(shape=(2,))  # [return, VaR]
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, output)

def train_gan():
    # Load and preprocess data
    tail_data, scaler = load_and_preprocess()
    
    # Build models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Combined model
    noise = tf.keras.Input(shape=(LATENT_DIM,))
    cond = tf.keras.Input(shape=(1,))
    fake_return = generator(tf.keras.layers.Concatenate()([noise, cond]))
    discriminator.trainable = False
    validity = discriminator(tf.keras.layers.Concatenate()([fake_return, cond]))
    combined = tf.keras.Model([noise, cond], validity)
    
    # Compile models
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    combined.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Training
    real_samples = tail_data
    cond_var = tail_data[:, 1:2]  # VaR values as condition
    
    for epoch in range(EPOCHS):
        # Train discriminator
        idx = np.random.randint(0, len(real_samples), BATCH_SIZE)
        real_batch = real_samples[idx]
        
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_returns = generator.predict(
            np.concatenate([noise, cond_var[idx]], axis=1),
            verbose=0
        )
        fake_batch = np.concatenate([fake_returns, cond_var[idx]], axis=1)
        
        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        g_loss = combined.train_on_batch(
            [noise, cond_var[idx]],
            np.ones((BATCH_SIZE, 1))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
    
    # Save models
    generator.save(SAVE_PATH)
    return generator, scaler

if __name__ == "__main__":
    train_gan()