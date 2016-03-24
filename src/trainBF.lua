------------------------------------------------------
-- Online Data Association using RNNs
-- 
-- A. Milan, S. H. Rezatofighi, K. Schindler, A. Dick, I. Reid
-- arxiv 2016
--
-- This file implements training of a Bayesian filter (BF)
--
-- @author Anton Milan (anton.milan@adelaide.edu.au)
--
-- This code is based on A. Karpathy's character-level
-- RNN https://github.com/karpathy/char-rnn
------------------------------------------------------


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'lfs'
require 'image'

require 'util.misc'   -- miscellaneous
require 'auxBF'       -- Bayesian Filtering specific auxiliary methods

--nngraph.setDebug(true)  -- uncomment for debug mode

local RNN = require 'model.RNNBF' -- RNN model for BF


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a simple trajectory model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-config', '', 'config file')
cmd:option('-rnn_size', 100, 'size of RNN internal state')
cmd:option('-num_layers',1,'number of layers in the RNN / LSTM')
cmd:option('-model_index',1,'1=lstm, 2=gru, 3=rnn')
cmd:option('-temp_win',10, 'temporal window history')
--cmd:option('-batch_size',1,'number of frames to consider (1=online)')
cmd:option('-max_n',3,'Max number of targets per frame')
cmd:option('-max_m',3,'Max number of measurements per frame')
cmd:option('-max_nf',3,'Max number of false targets per frame')
cmd:option('-state_dim',1,'state dimension (1-4)')
--cmd:option('-loss_type',1,'1 = loss(x,gt), 0 = loss(x, det)')
cmd:option('-order_dets',1,'order detections (HA) before feeding')
cmd:option('-kappa',5,'gt loss weighting')
cmd:option('-lambda',5,'pred loss weighting')
cmd:option('-mu',1,'label loss weighting')
cmd:option('-nu',1,'ex loss weighting')
cmd:option('-xi',1,'smoothness termination')
--cmd:option('-omicron',1,'dynamic smoothness')
--cmd:option('-vel',0,'include velocity into state')
--cmd:option('-linp',1,'use labels inputs')
--cmd:option('-einp',0,'use labels inputs')
--cmd:option('-one_hot',1,'use one hot encoding')
-- optimization
cmd:option('-lrng_rate',1e-2,'learning rate')
cmd:option('-lrng_rate_decay',0.99,'learning rate decay')
cmd:option('-lrng_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.97,'decay rate for rmsprop')
cmd:option('-dropout',0.02,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_epochs',10000,'number of full passes through the training data')
cmd:option('-grad_clip',.1,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-training_file', '', 'training data')
cmd:option('-mini_batch_size',10,'mini-batch size')
cmd:option('-rand_par_rng',0.1,'random parameter range')
cmd:option('-forget_bias',1,'forget get biases')
cmd:option('-use_gt_input', 0, 'use gt as current input at t')
cmd:option('-use_dagt_input', 1, 'use data association input (0=hungarian, 1=GT, 2=predicted)')
cmd:option('-full_set', 1, 'use full (or reduced [0]) training / val set')
cmd:option('-random_epoch', 0, 'random training data each epoch')
-- data related
cmd:option('-synth_training',100,'number of synthetic scenes to augment training data with')
cmd:option('-pert_training',0,'number of perturbed scenes to augment training data with')
cmd:option('-synth_valid',10,'number of synthetic scenes to augment validation data with')
cmd:option('-real_data', 0, 'use (1) or don\'t use (0) real data for validation')
cmd:option('-real_dets', 0, 'use (1) or don\'t use (0) real detections')
cmd:option('-det_noise', 0.0, 'Synthetic detections noise level')
cmd:option('-det_fail', 0.0, 'Synthetic detections failure rate')
cmd:option('-det_false', 0.0, 'Synthetic detections false alarm rate')
cmd:option('-det_thr', 0.03, 'detection to gt assignment threshold')
cmd:option('-reshuffle_dets', 1, 'reshuffle detections')
cmd:option('-fixed_n', 0, 'always same number of tracks')
cmd:option('-trim_tracks', 1, 'allow late birth and early death')
cmd:option('-dummy_det_val', 0, 'value for inexisting det. state')
cmd:option('-dummy_weight', 1, 'distance to dummy (missed) det')
cmd:option('-norm_std', 0, 'divide data by this factor (0=std.dev)')
cmd:option('-norm_mean', 1, 'set mean-zero')
cmd:option('-freq_reshuffle', 1, 'reshuffle detections each time')
cmd:option('-ex_thr', 0.5, 'theshold for target existence')
-- bookkeeping
cmd:option('-seed',122,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/mini-batches between printing out the loss')
cmd:option('-plot_every',10,'how many steps/mini-batches between plotting training')
cmd:option('-eval_val_every',100,'every how many iterations should we evaluate on validation data?')
cmd:option('-save_plots',0,'save plots as png')
cmd:option('-profiler',0,'profiler on/off')
cmd:option('-verbose',2,'Verbosity level')
cmd:option('-suppress_x',0,'suppress plotting in terminal')
cmd:option('-eval_mot15',1,'evaluate MOT15 with matlab')
cmd:option('-eval_conf','','evaluation config file')
cmd:option('-da_model','0305DA','model name for Data Association')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()


-- parse input params
opt = cmd:parse(arg)
-- overwrite params with config if given
if string.len(opt.config) > 0 then opt = parseConfig(opt.config, opt) end



------------------------------------------------------------------------
-- GLOBAL VARS for easier handling
stateVel = false
updLoss = opt.kappa ~= 0
predLoss = opt.lambda ~= 0
exVar = opt.nu ~= 0
smoothVar = opt.xi ~= 0
miniBatchSize = opt.mini_batch_size
stateDim = opt.state_dim
fullStateDim = stateDim if stateVel then fullStateDim = stateDim * 2 end
pm('State dimension:\t'..stateDim)
print('Full State dimension:\t'..fullStateDim)

checkCuda()       -- check for cuda availability
torch.manualSeed(opt.seed)  -- manual seed for deterministic runs
-----------------------------------------------------------------------
-- all sorts of options checking
if opt.real_data~=0 and opt.state_dim<2 then error('ERROR: Real Data requires dim >= 2') end
if opt.real_data==0 then opt.real_dets = 0 end  -- synthetic trajectories require synth. detections
if opt.real_dets ~= 0 then opt.det_noise, opt.det_fail, opt.det_false = 0,0,0 end -- no noise on real detections

-- we need at least N detections slots
if opt.max_m < opt.max_n then opt.max_m = opt.max_n + 0 end
local val_temp_win = opt.temp_win  -- change this to have a different validation window length

-- set model type (LSTM, GRU, RNN)
opt.model = 'lstm'
if opt.model_index==2 then opt.model='gru'
elseif opt.model_index==3 then opt.model='rnn' end

-- number of hidden inputs (1 for RNN, 2 for LSTM)
opt.nHiddenInputs = 1
if opt.model=='lstm' then opt.nHiddenInputs = 2 end

-- set network output indices
-- UPDATE
if updLoss then opt.updIndex = opt.num_layers*opt.nHiddenInputs+1 end
if opt.model=='lstm' then  opt.updIndex = 2*opt.num_layers+1 end
-- PREDICTION
opt.predIndex = opt.updIndex+1
if exVar then opt.exPredIndex = opt.predIndex+1 end   -- binary indicator
-- EXISTENCE AND EX. SMOOTHNESS
opt.exSmoothPredIndex = opt.predIndex+1
opt.dynSmoothPredIndex = opt.exSmoothPredIndex + 1

print('state update index\t'..opt.updIndex)
print('state prdctn index\t'..opt.predIndex)
if exVar then print('ex index\t\t'..opt.exPredIndex) end
if smoothVar then print('SmEx index\t\t'..opt.dynSmoothPredIndex) end

------------------------------------------------------------
------------------------------
----- BUILDING THE MODEL -----
------------------------------
modelName = 'default'   -- base name
if string.len(opt.config)>0 then local fp,fn,fe = fileparts(opt.config); modelName = fn end

-- a list of model-specific parameters... These cannot change across models
opt.modelParams = {'model_index', 'rnn_size', 'num_layers','max_n','max_m','state_dim'}
opt.dataParams = {'synth_training','synth_valid','mini_batch_size',
    'max_n','max_m','state_dim','full_set','fixed_n',
    'temp_win','real_data','real_dets','trim_tracks'}
    