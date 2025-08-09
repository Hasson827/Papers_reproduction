import matplotlib.pyplot as plt
import numpy as np

def visualization(original_data: np.ndarray, generated_data: np.ndarray, losses: list, model_name: str):
    plt.figure(figsize=(15, 5))

    # The original Distribution
    plt.subplot(1, 3, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=1, c='blue')
    plt.title('Original Distribution')
    plt.grid(True, alpha=0.3)
    
    # The generated Distribution
    plt.subplot(1, 3, 2)
    plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.6, s=1, c='red')
    plt.title('Generated Distributions')
    plt.grid(True, alpha=0.3)

    # The Train Loss Curve
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name} results.png', dpi=300, bbox_inches='tight')