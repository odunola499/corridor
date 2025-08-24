import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
from nemo.collections.asr.modules import ConformerMultiLayerFeatureExtractor


class AudioFeatureExtractor:
    def __init__(self, model_path: str = 'nvidia/ssl_en_nest_large_v1.0', device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = EncDecDenoiseMaskedTokenPredModel.restore_from('/home/odunola/Documents/papers/current_projects/rnnt/tests/ssl_en_nest_large_v1.0.nemo')
        self.model.to(self.device)
        self.preprocessor = self.model.preprocessor
        self.encoder = self.model.encoder
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(self.encoder)
        self.sample_rate = self.model.cfg.sample_rate
        
    def extract_features(self, audio: torch.Tensor, layers: Optional[List[int]] = None) -> np.ndarray:
        with torch.inference_mode():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            audio = audio.to(self.device)
            audio_length = torch.tensor([audio.shape[-1]], device=self.device)
            
            processed_signal, processed_length = self.preprocessor(
                input_signal=audio,
                length=audio_length
            )
            print(processed_signal.shape)
            print(processed_length)
            
            if layers:
                self.feature_extractor.layer_idx_list = layers
            
            features, _ = self.feature_extractor(
                audio_signal=processed_signal, 
                length=processed_length
            )
            
            return torch.stack(features, dim=0).cpu().numpy()
    
    def extract_from_file(self, audio_path: Union[str, Path], layers: Optional[List[int]] = None) -> np.ndarray:
        import librosa
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio_tensor = torch.from_numpy(audio).float()
        return self.extract_features(audio_tensor, layers)
    
    def batch_extract(self, audio_batch: torch.Tensor, layers: Optional[List[int]] = None) -> np.ndarray:
        with torch.inference_mode():
            audio_batch = audio_batch.to(self.device)
            batch_size = audio_batch.shape[0]
            audio_lengths = torch.full((batch_size,), audio_batch.shape[-1], device=self.device)
            
            processed_signal, processed_lengths = self.preprocessor(
                input_signal=audio_batch,
                length=audio_lengths
            )
            
            if layers:
                self.feature_extractor.layer_idx_list = layers
            
            features, _ = self.feature_extractor(
                audio_signal=processed_signal, 
                length=processed_lengths
            )
            
            return torch.stack(features, dim=1).cpu().numpy()


if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    
    test_audio = torch.randn(2, 32000)
    features = extractor.extract_features(test_audio)
    print(f"Features shape: {features.shape}")
    
    batch_features = extractor.batch_extract(torch.randn(4, 32000))
    print(f"Batch features shape: {batch_features.shape}")