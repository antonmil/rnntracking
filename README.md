# Online Multi-Target Tracking with Recurrent Neural Networks

This code accompanies the following paper: [(pdf)](http://www.milanton.de/files/aaai2017/aaai2017-anton-rnntracking.pdf)


    Online Multi-target Tracking Using Recurrent Neural Networks
    A. Milan, S. H. Rezatofighi, A. Dick, I. Reid, K. Schindler. In: AAAI 2017
    
bibtex:
```
@inproceedings{Milan:2017:AAAI_RNNTracking,
	title = {Online Multi-Target Tracking using Recurrent Neural Networks},
	booktitle = {AAAI},
	author = {Milan, A. and Rezatofighi, S. H. and Dick, A. and Reid, I. and Schindler, K.},
	month = {February},
	year = {2017}
}
```


# Dependencies
## Required
    * Lua
    * Torch
    * luarocks: nn, nngraph, lfs
    
## Optional    
    * cunn, cutorch (optional for GPU usage)
    * Matlab (visualization and metric computation only)

    
# Installation
## Torch
    # Follow these instructions to install Torch: http://torch.ch/docs/getting-started.html

    # in a terminal, run the commands
    curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
    git clone https://github.com/torch/distro.git ~/torch --recursive
    cd ~/torch; ./install.sh


## Additional Dependencies
### Luarocks
    sudo apt-get install luarocks
    luarocks install nn
    luarocks install nngraph
    luarocks install luafilesystem
    luarocks install https://raw.github.com/jucor/torch-distributions/master/distributions-0-0.rockspec
    luarocks install cutorch
    luarocks install cunn

### MOTChallenge
The current version uses the training set of the
[MOTChallenge benchmark](https://motchallenge.net) to generate synthetic data for training.

You should download the dataset from [here](https://motchallenge.net/data/2D_MOT_2015/) and
set the local path in
    src/util/paths.lua

    

# Usage


## Testing
The code comes with a pre-trained model located in `bin/rnnTracker.t7`. It was trained
on a subset of the [MOTChallenge 2015 training set](https://motchallenge.net/data/2D_MOT_2015/)

Run
    th rnnTracker.lua
to get a sense of the result on synthetic data or
    th rnnTracker.lua -model_name rnnTracker -seq_name TUD-Campus
    
to produce results on the `TUD-Campus` sequence. The bounding
boxes are saved in `./out/rnnTrack/TUD-Campus.txt`. Type
    th rnnTracker.lua -h to get a full list of options
    
This example uses Hungarian data association.

## Visualization
To see the visual results you can run
    th visBoxes.lua -file ../out/rnnTracker_r300_l1_n1_m1_d4/TUD-Campus.txt
    

## Training

    th trainBF.lua -config ../config/configBF.txt
will start training a model on the `TUD-Campus` sequence. Type
    th trainBF.lua -h

to see the full set of options. You may define the training parameters
in a separate text file, similar to `config/configBF.txt` and pass it
as the `-config` option to the training script.



## Data
Training expects annotated image sequences. The annotation format is a CSV text file
following the syntax of the [MOTChallenge benchmark](https://motchallenge.net).
For testing, the image sequence and the corresponding set of detection in the same
format is required.

### Internal representation
Internally, all data (tracks and detections) is stored in N x F x D tensors, where 

* N = max. number of targets / detections
* F = number of frames in a batch
* D = dimensionality (e.g. 2 for (x,y) or 4 for (x,y,w,h)

The labels (data association) is represented by an NxF tensor.

Furthermore, training, validation and real-data sets are kept in a lua table.
I.e. each entry in a table is then a MB*N x F x D tensor, where MB is the mini-batch size. There are four tables for each set.

1. Tracks
2. Detections
3. Labels
4. Sequence names (used for generating that one specific datum)


# Documentation
The code is documented following the luadoc convention. To generate
html docs, install luadoc `luarocks install luadoc` and run `./docify.sh`.

# Known issues
## Data Association
The code for training data association is not included yet. We are working on releasing it soon.

## Training data
Training data is generated synthetically by learning simple generative trajectory models from annotated data. Training with real data is not supported.



# Acknowledgements and remarks

We thank Andrej Karpathy for releasing his 
[code](https://github.com/karpathy/char-rnn) on character-level
RNNs that served as basis for this project.

**Pull requests welcome.**




# License

BSD License