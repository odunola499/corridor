# Audio Encoder Workbench

A workbench for experimenting with audio encoders and their components.
This is actively being worked on.
## Components

### Core (`src/core.py`)
- **VectorQuantize**: self-supervised targets with Q seperate classifers and quantizers (following Assembly AI Universal 1 https://arxiv.org/html/2404.09841v2)
- **BaseTrainEngine**: Abstract training engine base class
- **VectorTrainEngine**: BestRQ algorithm implementation for pretraining audio encoders (used in Google USM and Assembly AI Universal 1)
- **mask_features**: Feature masking for self-supervised learning

### Configuration (`src/config.py`)
- **AudioConfig**, **TrainConfig**, **DataConfig**: Base configuration classes
- **ConformerConfig**, **WhisperConfig**: Audio encoder configurations
- **RVQTrainConfig**: BestRQ algorithm configuration

### Encoders (`src/encoder/`)
- **Conformer**: Convolution transformer
- **WhisperEncoder**: Whisper's audio encoder

### Decoders (`src/decoder/`)
- **RVQ**: BestRQ implementation
- **Whisper Experimental RQ**: Experimental BestRQ-style continued pretraining with Whisper's Encoder

### Feature Extractors
- Work in Progress

### Text Tokenizers
- Work in Progress