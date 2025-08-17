# Audio Encoder Workbench

A workbench for experimenting with audio encoders and their components.

## Components

### Core (`src/core.py`)
- **VectorQuantize**: Multi-layer vector quantization
- **BaseTrainEngine**: Abstract training engine base class
- **VectorTrainEngine**: BestRQ algorithm implementation for pretraining audio encoders (used in Google USM and Assembly AI Universal 1)
- **mask_features**: Feature masking for self-supervised learning

### Configuration (`src/config.py`)
- **AudioConfig**, **TrainConfig**, **DataConfig**: Base configuration classes
- **ConformerConfig**, **WhisperConfig**: Audio encoder configurations
- **RVQTrainConfig**: BestRQ algorithm configuration

### Encoders (`src/encoder/`)
- **Conformer**: Convolution-augmented transformer
- **WhisperEncoder**: Whisper-based audio encoder
- Layer components: attention, FFN, convolution, specaugment

### Decoders (`src/decoder/`)
- **RVQ**: BestRQ implementation
- **Whisper Experimental RQ**: Experimental BestRQ-style continued pretraining with Whisper

### Feature Extractors

### Text Tokenizers