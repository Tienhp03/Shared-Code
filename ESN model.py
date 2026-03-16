import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time

class ESNChannelPredictor:
    """
    Echo State Network for FSO Channel Prediction
    Based on equations (9)-(15) in the paper
    """
    
    def __init__(self, input_size=20, reservoir_size=100, output_size=1, 
                 leaking_rate=0.2, spectral_radius=0.8, sparsity=0.1,
                 regularization_coef=1e-4, random_seed=42):
        """
        Initialize ESN model với parameters tối ưu hơn
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.regularization_coef = regularization_coef
        
        np.random.seed(random_seed)
        
        # Initialize weights
        self._initialize_weights()
        
        # Reservoir state
        self.state = np.zeros(reservoir_size)
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Training info
        self.training_time = 0
        self.is_trained = False
        
    def _initialize_weights(self):
        """Initialize weights correctly"""
        # Input weights: small random values
        self.W_in = np.random.randn(self.reservoir_size, self.input_size + 1) * 0.1
        
        # Reservoir weights: sparse random
        self.W = np.random.randn(self.reservoir_size, self.reservoir_size) * 0.1
        
        # Apply sparsity
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        self.W *= mask
        
        # Scale spectral radius
        eigenvalues = np.abs(np.linalg.eigvals(self.W))
        if np.max(eigenvalues) > 0:
            self.W *= self.spectral_radius / np.max(eigenvalues)
        
        self.W_out = None
        
    def _update_state(self, input_data):
        """Update reservoir state"""
        # Concatenate bias and input: [1; x(n)]
        u = np.concatenate([[1], input_data])
        
        # Compute updated state
        r_bar = np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.state))
        
        # Update state with leaking rate
        self.state = (1 - self.leaking_rate) * self.state + \
                     self.leaking_rate * r_bar
        
        return self.state
    
    def create_training_data(self, channel_data, M=20, N=1):
        """Create training data với sliding window"""
        X, Y = [], []
        n_samples = len(channel_data)
        
        for i in range(n_samples - M - N + 1):
            X.append(channel_data[i:i+M])
            Y.append(channel_data[i+M:i+M+N])
            
        return np.array(X), np.array(Y)
    
    def train(self, X_train, Y_train, warmup_samples=200):
        """Train ESN model với ridge regression"""
        start_time = time.time()
        n_samples = len(X_train)
        
        print(f"Training ESN với {n_samples} samples...")
        
        # Reset state
        self.state = np.zeros(self.reservoir_size)
        
        # Warmup phase
        for i in range(min(warmup_samples, n_samples)):
            self._update_state(X_train[i])
        
        # Collect states
        states = []
        for i in range(warmup_samples, n_samples):
            state = self._update_state(X_train[i])
            Z = np.concatenate([[1], X_train[i], state])
            states.append(Z)
        
        # Build matrices
        X_matrix = np.array(states).T
        Y_target = Y_train[warmup_samples:].T
        
        if len(Y_target.shape) == 1:
            Y_target = Y_target.reshape(1, -1)
        
        # Ridge regression
        XTX = np.dot(X_matrix, X_matrix.T)
        regularization = self.regularization_coef * np.eye(XTX.shape[0])
        
        try:
            self.W_out = np.dot(np.dot(Y_target, X_matrix.T), 
                               np.linalg.inv(XTX + regularization))
            self.is_trained = True
        except:
            self.W_out = np.dot(np.dot(Y_target, X_matrix.T), 
                               np.linalg.pinv(XTX + regularization))
            self.is_trained = True
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
    def predict(self, X_test):
        """Predict future values"""
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
        
        predictions = []
        self.state = np.zeros(self.reservoir_size)  # Reset state cho test
        
        for x in X_test:
            state = self._update_state(x)
            Z = np.concatenate([[1], x, state])
            y_pred = np.dot(self.W_out, Z)
            predictions.append(y_pred)
            
        return np.array(predictions)
    
    def normalize_data(self, data):
        """Normalize data"""
        return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def denormalize_data(self, data):
        """Denormalize data"""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def generate_realistic_fso_channel(n_samples=10000, turbulence_strength=0.1):
    """
    Tạo FSO channel data thực tế hơn với temporal correlation
    """
    print(f"Generating FSO channel with σ_R² = {turbulence_strength}...")
    
    # Time vector
    t = np.linspace(0, 10, n_samples)
    
    # 1. Atmospheric attenuation (slow variations)
    h_l = 0.5 + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    
    # 2. Turbulence (lognormal with temporal correlation)
    sigma_R = np.sqrt(turbulence_strength)
    
    # Tạo noise có correlation
    n = len(t)
    correlation_length = 100  # Samples
    correlation = np.exp(-np.arange(n) / correlation_length)
    correlation = correlation[:n//2]
    correlation = np.concatenate([correlation[::-1], correlation[1:]])
    
    # Tạo correlated noise
    noise = np.random.randn(n)
    noise_correlated = np.convolve(noise, correlation[:100], mode='same')
    noise_correlated = noise_correlated / np.std(noise_correlated)
    
    # Transform to lognormal
    log_normal = np.exp(sigma_R * noise_correlated - sigma_R**2/2)
    
    # 3. Pointing errors
    pointing = 0.9 + 0.1 * np.sin(2 * np.pi * 0.2 * t) + 0.05 * np.random.randn(n)
    pointing = np.clip(pointing, 0.7, 1.0)
    
    # Composite channel
    h = h_l * log_normal * pointing
    
    # Add small noise
    h_noisy = h + 0.01 * np.random.randn(n)
    
    return h_noisy, h


def evaluate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    data_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / data_range if data_range > 0 else 0
    
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'Correlation': correlation
    }


def plot_detailed_results(y_true, y_pred, title="ESN Prediction Results"):
    """Plot detailed results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Time series
    ax1 = axes[0, 0]
    n_plot = 300
    ax1.plot(y_true[:n_plot], 'b-', label='Actual', alpha=0.7)
    ax1.plot(y_pred[:n_plot], 'r--', label='Predicted', alpha=0.7)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Channel Gain')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=1)
    ax2.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 'r--')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = axes[0, 2]
    errors = y_true - y_pred
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution (μ={errors.mean():.4f}, σ={errors.std():.4f})')
    ax3.grid(True, alpha=0.3)
    
    # Autocorrelation của actual
    ax4 = axes[1, 0]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(y_true[:1000], lags=50, ax=ax4, alpha=0.05)
    ax4.set_title('Autocorrelation (Actual)')
    ax4.grid(True, alpha=0.3)
    
    # Autocorrelation của error
    ax5 = axes[1, 1]
    plot_acf(errors[:1000], lags=50, ax=ax5, alpha=0.05)
    ax5.set_title('Autocorrelation (Error)')
    ax5.grid(True, alpha=0.3)
    
    # Prediction vs Actual (zoomed)
    ax6 = axes[1, 2]
    ax6.plot(y_true[:100], 'b-', label='Actual', alpha=0.7, linewidth=2)
    ax6.plot(y_pred[:100], 'r--', label='Predicted', alpha=0.7, linewidth=2)
    ax6.set_xlabel('Sample')
    ax6.set_ylabel('Channel Gain')
    ax6.set_title('Zoomed View (100 samples)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main function với parameters đã sửa"""
    print("="*70)
    print("FSO CHANNEL PREDICTION USING ESN (FIXED VERSION)")
    print("="*70)
    
    # Parameters
    M = 20          # Input size
    N = 1           # Output size
    n_samples = 10000
    
    # Test với 2 turbulence levels
    turbulence_levels = [0.0075, 0.1020]
    
    for turb in turbulence_levels:
        print(f"\n{'-'*70}")
        print(f"TURBULENCE STRENGTH: σ_R² = {turb}")
        print(f"{'-'*70}")
        
        # Generate data
        channel_data, true_channel = generate_realistic_fso_channel(
            n_samples=n_samples,
            turbulence_strength=turb
        )
        
        # Create ESN với parameters tối ưu hơn
        esn = ESNChannelPredictor(
            input_size=M,
            reservoir_size=200,        # Tăng lên 200
            output_size=N,
            leaking_rate=0.2,           # Giảm xuống 0.2
            spectral_radius=0.8,        # Giảm xuống 0.8
            sparsity=0.1,
            regularization_coef=1e-4,    # Tăng lên 1e-4
            random_seed=42
        )
        
        # Normalize
        channel_norm = esn.normalize_data(channel_data)
        
        # Create training data
        X, Y = esn.create_training_data(channel_norm, M=M, N=N)
        
        # Split
        train_size = 7000
        test_size = 2000
        
        X_train, X_test = X[:train_size], X[train_size:train_size+test_size]
        Y_train, Y_test = Y[:train_size], Y[train_size:train_size+test_size]
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train
        esn.train(X_train, Y_train, warmup_samples=300)
        
        # Predict
        Y_pred = esn.predict(X_test)
        
        # Denormalize
        Y_test_denorm = esn.denormalize_data(Y_test.reshape(-1, 1)).flatten()
        Y_pred_denorm = esn.denormalize_data(Y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = evaluate_metrics(Y_test_denorm, Y_pred_denorm)
        
        print(f"\nKẾT QUẢ DỰ ĐOÁN:")
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"NRMSE: {metrics['NRMSE']:.6f}")
        print(f"Correlation: {metrics['Correlation']:.4f}")
        
        # Plot
        plot_detailed_results(Y_test_denorm, Y_pred_denorm, 
                             f"ESN Results (σ_R² = {turb})")
        
        # Đánh giá nhanh
        print(f"\nĐÁNH GIÁ:")
        if metrics['Correlation'] > 0.8:
            print("✅ RẤT TỐT: Model học được pattern")
        elif metrics['Correlation'] > 0.5:
            print("⚠️ TẠM ĐƯỢC: Có thể cải thiện thêm")
        else:
            print("❌ KÉM: Model chưa học được pattern")
        
        if metrics['NRMSE'] < 0.1:
            print("✅ Sai số nhỏ (<10%)")
        elif metrics['NRMSE'] < 0.2:
            print("⚠️ Sai số trung bình (10-20%)")
        else:
            print("❌ Sai số lớn (>20%)")


if __name__ == "__main__":
    main()