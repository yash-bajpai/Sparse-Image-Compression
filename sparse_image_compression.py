import numpy as np
from scipy import sparse
from scipy.fftpack import dct, idct
from PIL import Image
import os

class SparseImageCompressor:
    def __init__(self, compression_ratio=0.3, block_size=8):
        """
        Initialize the sparse image compressor.
        
        Args:
            compression_ratio (float): Ratio of non-zero elements to keep (0-1)
            block_size (int): Size of blocks for DCT transformation
        """
        self.compression_ratio = compression_ratio
        self.block_size = block_size
        
    def _image_to_matrix(self, image):
        """Convert PIL Image to numpy array."""
        return np.array(image)
    
    def _matrix_to_image(self, matrix):
        """Convert numpy array to PIL Image."""
        # Ensure the values are in valid range
        matrix = np.clip(matrix, 0, 255)
        return Image.fromarray(np.uint8(matrix))
    
    def _apply_dct(self, channel):
        """Apply DCT to a channel in blocks."""
        height, width = channel.shape
        dct_channel = np.zeros_like(channel, dtype=np.float32)
        
        # Apply DCT to each block
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = channel[i:i+self.block_size, j:j+self.block_size]
                if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    dct_channel[i:i+self.block_size, j:j+self.block_size] = dct_block
        
        return dct_channel
    
    def _apply_idct(self, dct_channel):
        """Apply inverse DCT to reconstruct the channel."""
        height, width = dct_channel.shape
        channel = np.zeros_like(dct_channel, dtype=np.float32)
        
        # Apply IDCT to each block
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = dct_channel[i:i+self.block_size, j:j+self.block_size]
                if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                    idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
                    channel[i:i+self.block_size, j:j+self.block_size] = idct_block
        
        return channel
    
    def _compress_channel(self, channel):
        """Compress a single channel using DCT and sparse matrix operations."""
        # Apply DCT
        dct_channel = self._apply_dct(channel)
        
        # Flatten the DCT coefficients
        flat_dct = dct_channel.flatten()
        
        # Calculate number of elements to keep
        n_elements = len(flat_dct)
        k = int(n_elements * self.compression_ratio)
        
        # Get indices of top k elements by magnitude
        ind = np.argpartition(np.abs(flat_dct), -k)[-k:]
        
        # Create sparse array
        sparse_dct = np.zeros_like(flat_dct)
        sparse_dct[ind] = flat_dct[ind]
        
        # Reshape back to original dimensions
        sparse_dct = sparse_dct.reshape(dct_channel.shape)
        
        # Apply inverse DCT
        compressed_channel = self._apply_idct(sparse_dct)
        
        return compressed_channel
    
    def compress_image(self, input_path, output_path):
        """
        Compress an image using DCT and sparse matrix operations.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save compressed image
        """
        # Load image
        image = Image.open(input_path)
        matrix = self._image_to_matrix(image)
        
        # Handle both grayscale and color images
        if len(matrix.shape) == 2:  # Grayscale
            compressed_matrix = self._compress_channel(matrix)
        else:  # Color
            # Process each color channel
            compressed_channels = []
            for channel in range(matrix.shape[2]):
                channel_matrix = matrix[:, :, channel]
                compressed_channel = self._compress_channel(channel_matrix)
                compressed_channels.append(compressed_channel)
            
            # Combine channels
            compressed_matrix = np.stack(compressed_channels, axis=2)
        
        # Save compressed image
        compressed_image = self._matrix_to_image(compressed_matrix)
        compressed_image.save(output_path)
        
        # Calculate compression statistics
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = 1 - (compressed_size / original_size)
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        }

def main():
    #Example usage
    compressor = SparseImageCompressor(compression_ratio=0.1, block_size=8)
    
    #Using existing image2.jpg as input
    input_image = "image2.jpg"
    output_image = "compressed_output.jpg"
    
    if os.path.exists(input_image):
        stats = compressor.compress_image(input_image, output_image)
        print(f"Original size: {stats['original_size']} bytes")
        print(f"Compressed size: {stats['compressed_size']} bytes")
        print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    else:
        print(f"Error: Input image '{input_image}' not found")

if __name__ == "__main__":
    main() 