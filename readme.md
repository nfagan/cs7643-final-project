This repository contains the code accompanying my replication experiment of Huang et al.'s Music Transformer [1]. This code was developed with Pytorch version 1.13 and CUDA version 11.6, although previous versions of these may work. This code additionally depends on numpy, scipy, and [Music21](http://web.mit.edu/music21/) for MIDI file processing.

# Install

With conda: create a new environment with Python version >= 3.9 and activate it. Follow the [instructions](https://pytorch.org/) to install Pytorch for your platform. Then install the remaining dependencies: `conda install numpy scipy` and `pip install music21`.

# Train

`python experiment/train.py` trains a model configuration that depends on the values of flags set in `train.py`.
* `IS_RNN`, if true, will train an LSTM-based recurrent model due to Oore et al. [2]; otherwise a Transformer decoder will be trained.
* `IS_JSB`, if true, uses note sequences from Bach chorales for the training set; otherwise, note sequences from the Maestro dataset are used.
* `IS_RPR`, if true, follows Huang et al. in incorporating embeddings of relative sequence position prior to calculation of the self-attention weights; otherwise, this embedding is disabled.
* `DISABLE_POS_EMBED`, if true, disables embeddings of absolute sequence position in the input to the decoder.

# Generate

`python experiment/generate.py` loads a trained model checkpoint to generate conditioned or uncoditioned MIDI note sequences.
