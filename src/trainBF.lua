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

nngraph.setDebug(true)  -- uncomment for debug mode
torch.setdefaulttensortype('torch.FloatTensor')

local RNN = require 'model.RNNBF' -- RNN model for BF
local model_utils = require 'util.model'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a simple trajectory model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-config', '', 'config file')
cmd:option('-rnn_size', 100, 'size of RNN internal state')
cmd:option('-num_layers',1,'number of layers in the RNN / LSTM')
cmd:option('-model_index',3,'1=lstm, 2=gru, 3=rnn')
cmd:option('-temp_win',10, 'temporal window history')
--cmd:option('-batch_size',1,'number of frames to consider (1=online)')
cmd:option('-max_n',1,'Max number of targets per frame')
cmd:option('-max_m',1,'Max number of measurements per frame')
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

-- create auxiliary directories (or make sure they exist)
createAuxDirs()


------------------------------------------------------------------------
-- GLOBAL VARS for easier handling
-- TODO: Replace with local or opt. members
stateVel = false
updLoss = opt.kappa ~= 0
predLoss = opt.lambda ~= 0
exVar = opt.nu ~= 0
smoothVar = opt.xi ~= 0
miniBatchSize = opt.mini_batch_size
stateDim = opt.state_dim
fullStateDim = stateDim if stateVel then fullStateDim = stateDim * 2 end
maxTargets,maxDets,nClasses = opt.max_n,opt.max_m,opt.max_m+1

xSize = stateDim*maxTargets
dSize = stateDim*maxDets
fullxSize = fullStateDim*maxTargets

opt.xSize, opt.dSize = xSize, dSize
opt.nClasses = opt.max_m+1

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

-- adjust sample size
opt.synth_training = math.max(miniBatchSize,opt.synth_training)
opt.synth_valid = math.max(miniBatchSize,opt.synth_valid)

-- adjust sample size
opt.synth_training = math.floor(opt.synth_training/miniBatchSize) * miniBatchSize
opt.synth_valid = math.floor(opt.synth_valid/miniBatchSize) * miniBatchSize

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

-- prototype for one time step
print("Memory Type: "..opt.model)
-- define the model: prototypes for one timestep, then cmodel_batchlone them in time
local do_random_init = true
local itOffset = 0
if string.len(opt.init_from) > 0 then
  print('loading an LSTM from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos

  -- need to adjust some crucial parameters according to checkpoint
  pm('overwriting ...',2)
  for _, v in pairs(modelParams) do
    if opt[v] ~= checkpoint.opt[v] then
      opt[v] = checkpoint.opt[v]
      pm(string.format('%15s',v) ..'\t = ' .. checkpoint.opt[v], 2)
    end
  end
  pm('            ... based on the checkpoint.',2)
  do_random_init = false
  itOffset = checkpoint.it
  pm('Resuming from iteration '..itOffset+1,2)

  --   opt.max_epochs = opt.max_epochs + itOffset
  --   pm('New max_epochs = '..opt.max_epochs)
else
  print('creating an '..opt.model..' with ' .. opt.num_layers .. ' layers')
  protos = {}
  if opt.model == 'lstm' then protos.rnn = LSTM.lstm(opt)
  elseif opt.model == 'gru' then protos.rnn = GRU.gru(opt)
  elseif opt.model == 'rnn' then protos.rnn = RNN.rnn(opt)
  else print('Unknown model. Take LSTM'); protos.rnn = LSTM.lstm(opt) end
  --   local msec = nn.MSECriterion()
  --   protos.criterion = nn.ParallelCriterion()
  --   protos.criterion:add(msec, opt.gt_loss)
  --   protos.criterion:add(msec, opt.det_loss)
  --   protos.criterion = nn.MSECriterion()
  local lambda = opt.lambda
  local nllc = nn.ClassNLLCriterion()
  local bce = nn.BCECriterion()
  local abserr = nn.AbsCriterion()
  local mserr = nn.MSECriterion()

  protos.criterion = nn.ParallelCriterion()
  if updLoss then protos.criterion:add(mserr, opt.kappa) end
  if predLoss then protos.criterion:add(mserr, opt.lambda) end
  if exVar then protos.criterion:add(bce, opt.nu) end
  if smoothVar then protos.criterion:add(mserr, opt.xi) end

end

---------------------
-- LOAD  DA  NETWORK
---------------------
if opt.use_da_input==2 then
-- TODO
else
  DAopt = deepcopy(opt)
end


-- the initial state of the cell/hidden states
init_state = getInitState(opt, miniBatchSize)
val_init_state = getInitState(opt, 1)
DAinit_state = getDAInitState(opt, miniBatchSize)
-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

if do_random_init then params:uniform(-opt.rand_par_rng, opt.rand_par_rng) end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to '..opt.forget_bias..' in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(opt.forget_bias)
      end
    end
  end
end
print('number of parameters in the model: ' .. params:nElement())

-- make a bunch of clones after flattening, as that reallocates memory
pm('Cloning '..opt.temp_win..' times...',2)
clones = {}
for name,proto in pairs(protos) do
  print('\tcloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.temp_win, not proto.parameters)
end
pm('   ...done',2)

-- plot network as graph
if not onNetwork() then  graph.dot(protos.rnn.fg, 'RNNBFF', './graph/RNNBFForwardGraph') end

---------------------------------------------------
----- now comes data generation ---
-- just take MOTChallenge 2DMOT2015 training set
-- Note: I know, it may be better to split into training/validation
-- but didn't make much difference in practice
local trSeqTable={'TUD-Campus','TUD-Stadtmitte','PETS09-S2L1','ETH-Sunnyday',
  'ETH-Bahnhof','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8','Venice-2',
  'KITTI-13','KITTI-17'}
trSeqTable={'TUD-Campus'} -- for quick testing
local valSeqTable=trSeqTable


pm('Training Data...')
pm(trSeqTable)

pm(string.format('reading data...'),2)
trTracksTab, trDetsTab = getTracksAndDetsTables(trSeqTable, maxTargets, maxDets, nil, false)
pm(string.format('   ...done'),2)
pm('Validation Data...')
pm(valSeqTable)

-- Put image sizes in a table. Needed for normalization
imSizes = getImSizes(trSeqTable)
imSizes = getImSizes(valSeqTable, imSizes)

-- GET DATA (Synthetic) ---
local trTracksTab, trDetsTab, trLabTab, trExTab, trDetExTab, trSeqNames = prepareData('train', trSeqTable, trSeqTable, false)
-- local tt=opt.trim_tracks opt.trim_tracks=1
local tmpOpt=deepcopy(opt) -- do some modificiations for validation
opt.mini_batch_size = 1
miniBatchSize = opt.mini_batch_size
opt.temp_win = val_temp_win
local valTracksTab, valDetsTab, valLabTab, valExTab, valDetExTab, valSeqNames = prepareData('validation', valSeqTable, trSeqTable,  false)
opt = deepcopy(tmpOpt) -- restore
miniBatchSize = opt.mini_batch_size

print('Training batches:   '..tabLen(trTracksTab))
print('Validation batches: '..tabLen(valTracksTab))

--print(trSeqNames)
--showData(trTracksTab, trDetsTab, trSeqNames)
-------------------------------------------------------------------------

--------------------------------------------------------------------------
--- Get ground truth locations for frame t+1
-- @param t   current frame
function getGTState(t)
  return tracks[{{},{t+1}}]:reshape(miniBatchSize,fullxSize):float()
end

--------------------------------------------------------------------------
--- Get ground truth existence labels for frame t+1
-- @param t   current frame
function getGTEx(t)
  return exlabels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets):float()
end

--------------------------------------------------------------------------
--- Get ground truth labels (data association) for next frame
-- @param t   current frame
function getGTDA(t)
  local DA = labels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
  DA = dataToGPU(DA)
  return DA
end

--------------------------------------------------------------------------
--- Display and print result at current training iteration
function plotProgress(state, detections, predTracks, predDA, predEx, winID, winTitle,predTracks2)
  local loc = state:clone()
  if globiter == nil then globiter = 0 end
  if itOffset == nil then itOffset = 0 end
  predDA = predDA:sub(1,maxTargets):float()
  local predExO = predEx:clone()
  if exVar then predEx = predEx:sub(1,maxTargets):float() else predEx = nil end

  local DA = getLabelsFromLL(predDA, false)

  local plotTab={}
  local N,F,D=getDataSize(detections:sub(1,maxDets))
  local da = torch.IntTensor(N,F)
  if colorDetsShuffled then
    for i=1,maxDets do da[i]=(torch.ones(1,opt.temp_win)*i) end -- color dets shuffled
  end
  local fullDA = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), DA, 2)

  opt.plot_dim=1
  plotTab = getTrackPlotTab(tracks:sub(1,maxTargets):float(), plotTab, 1)
  plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, da)

  -- TODO REMOVE GLOBAL STATE
  GTSTATE = state:clone()
  plotTab = getDAPlotTab(predTracks:sub(1,maxTargets):float(), detections:sub(1,maxDets):float(), plotTab, fullDA, predEx,0,predDA)

  plotTab = getTrackPlotTab(predTracks:sub(1,maxTargets):float(), plotTab, 2, nil, predEx, 1) -- update
  plotTab = getTrackPlotTab(predTracks2:sub(1,maxTargets):float(), plotTab, 3, nil, predEx, 1, nil) -- prediction
  plotTab = getExPlotTab(plotTab, predExO, 1)
  plot(plotTab, winID, string.format('%s-%06d',winTitle,globiter+itOffset), nil, opt.save_plots) -- do not save first

  sleep(.01)
end

--------------------------------------------------------------------------
--- To compute loss and gradients, we construct tables with
-- predicted values (input) and ground truth values (target)
function getPredAndGTTables(stateUpd, statePred, predDA, predEx, smoothnessEx, smoothnessDyn, t)

  local input, target = {}, {}

  -- STATE FIRST
  local GTLocs = {}
  if updLoss or predLoss then GTLocs = getGTState(t)end

  -- UPDATE
  if updLoss then
    table.insert(input, stateUpd)  -- predicted (updated) state
    table.insert(target, GTLocs)  -- detection at predicted DA
  end

  -- PREDICTION
  if predLoss then
    table.insert(input, statePred)
    table.insert(target, GTLocs)
  end

  -- EXISTENCE
  if exVar then
    local exTar = getGTEx(t)
    table.insert(input, predEx)
    table.insert(target, exTar)
  end

  -- SMOOTHNESS TERMINATION
  if smoothVar then
    table.insert(input, smoothnessEx)
    table.insert(target, torch.zeros(maxTargets*miniBatchSize):float())
  end

  return input, target
end



-------------------------------------
--- LOSS AND GRADIENT COMPUTATION ---
-------------------------------------
local permPlot = math.random(tabLen(trTracksTab)) -- always plot this training seq

local seqCnt=0


function feval()
  grad_params:zero()
  tL = tabLen(trTracksTab)
  local randSeq = math.random(tL) -- pick a random sequence from training set
  seqCnt=seqCnt+1
  if seqCnt > tabLen(trTracksTab) then seqCnt = 1 end
  randSeq = seqCnt

  permPlot = math.random(tabLen(trTracksTab)) -- randomize plot
  if (globiter % opt.plot_every) == 0 then randSeq = permPlot end -- force same sequence for plotting

  -- These are global data for this one iteration
  tracks = trTracksTab[randSeq]:clone()
  detections = trDetsTab[randSeq]:clone()
  labels = trLabTab[randSeq]:clone()
  exlabels = trExTab[randSeq]:clone()
  detexlabels = trDetExTab[randSeq]:clone()


  ----- FORWARD ----
  local initStateGlobal = clone_list(init_state)
  local rnn_state = {[0] = initStateGlobal}
  local predictions = {}
  local predictionsTemp = {}

  local DAinitStateGlobal = clone_list(DAinit_state)
  local DArnn_state = {[0] = DAinitStateGlobal}
  local DApredictions = {}


  local loss = 0
  local stateLocs, stateLocs2, stateDA, stateEx = {}, {}, {}, {}
  local smoothnessEx = {}
  local smoothnessDyn = {}
  stateEx={[0]=torch.ones(maxTargets):float()}
  local statePred = {}
  local T = opt.temp_win - opt.batch_size
  local GTDA = {}

  local allPredDA = torch.zeros(maxTargets*miniBatchSize,T,nClasses)
  local allPredEx = torch.zeros(maxTargets*miniBatchSize, T, 1)


  TRAINING = true -- flag to be used for input

  DAtmpStateEx = torch.ones(maxTargets*miniBatchSize):float()
  DAtmpDA = torch.zeros(maxTargets*miniBatchSize, nClasses):float()
  DAtmpDAsum = torch.zeros(maxTargets*miniBatchSize, 1):float()

  for t=1,T do
    clones.rnn[t]:training()      -- set flag for dropout


    if opt.use_da_input==2 then error('TODO: train motion with predicted data association') end


    local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table

    for i=1,maxTargets*miniBatchSize do allPredDA[{{i},{t},{}}] = globalLAB[i]:clone() end -- copy current data association

    local lst = clones.rnn[t]:forward(rnninp) -- do one forward tick
    predictions[t] = lst

    -- update hidden state
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

    -- prediction at time t (for t+1)
    stateLocs[t], stateLocs2[t], stateDA[t], stateEx[t], smoothnessEx[t], smoothnessDyn[t] = decode(predictions, t)


    local input, target = getPredAndGTTables(stateLocs[t], stateLocs2[t], stateDA[t], stateEx[t], smoothnessEx[t], smoothnessDyn[t], t)

    local tloss = clones.criterion[t]:forward(input, target) -- compute loss for one frame
    loss = loss + tloss
  end
  loss = loss / T -- make average over all frames
  local predTracks, predTracks2, predDA, predEx = decode(predictions)

  -- plotting
  if (globiter == 1) or (globiter % opt.plot_every == 0) then
    local predDA = allPredDA:clone()
    predEx = predEx:sub(1,maxTargets)
    predDA = predDA:sub(1,maxTargets)

    local predLab = getLabelsFromLL(predDA, false) -- do not resolve with HA
    predLab = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), predLab, 2)

    feval_multass = printDA(predDA, predLab, predEx)

    plotProgress(tracks, detections, predTracks, predDA, predEx, 1, 'Training',predTracks2)
    feval_mm = getDAError(predDA, labels:sub(1,maxTargets))
  end

  ------ BACKWARD ------
  local rnn_backw = {}
  -- gradient at final frame is zero
  local drnn_state = {[T] = clone_list(init_state, true)} -- true = zero the clones
  for t=T,1,-1 do
    local input, target = getPredAndGTTables(stateLocs[t], stateLocs2[t], stateDA[t], stateEx[t], smoothnessEx[t], smoothnessDyn[t], t)

    local dl = clones.criterion[t]:backward(input,target)
    for dd=1,#dl do
      table.insert(drnn_state[t], dl[dd]) -- gradient of loss at time t
    end

    local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table

    local dlst = clones.rnn[t]:backward(rnninp, drnn_state[t])

    drnn_state[t-1] = {}
    local maxk = opt.num_layers+1
    if opt.model == 'lstm' then maxk = 2*opt.num_layers+1 end
    for k,v in pairs(dlst) do
      if k > 1 and k <= maxk then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the
        -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-1] = v
      end
    end
    -- TODO transfer final state?

    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  end

  return loss, grad_params

end


-------------------------------------------------------
-- OPTIMIZATION
-- start optimization here

train_losses = {}
val_losses = {}
real_losses = {}
train_mm, val_mm, real_mm = {}, {}, {}
mot15_mota = {}

train_lossesHA = {}
val_lossesHA = {}
real_lossesHA = {}
train_mmHA, val_mmHA, real_mmHA = {}, {}, {}
mot15_motaHA = {}
local optim_state = {learningRate = opt.lrng_rate, alpha = opt.decay_rate}
local glTimer = torch.Timer()
for i = 1, opt.max_epochs do
  local epoch = i
  globiter = i

  if i>1 and (i-1)%opt.synth_training==0 and opt.random_epoch~=0 then
    trTracksTab, trDetsTab, trLabTab, trSeqNames = prepareData('train', trSeqTable, trSeqTable, false)
  end

  local timer = torch.Timer()
  local _, loss = optim.rmsprop(feval, params, optim_state)
  local time = timer:time().real

  local train_loss = loss[1] -- the loss is inside a list, pop it
  train_losses[i] = train_loss

  -- exponential learning rate decay
  if i % (torch.round(opt.max_epochs/10)) == 0 and opt.lrng_rate_decay < 1 then
    --         if epoch >= opt.lrng_rate_decay_after then

    --       print('decreasing learning rate')
    local decay_factor = opt.lrng_rate_decay
    optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    --         end
  end

  -- print training progress
  if i % opt.print_every == 0 then
    printTrainingStats(i, opt.max_epochs, train_loss, grad_params:norm() / params:norm(),
      time, optim_state.learningRate, glTimer:time().real)
  end



  -- evaluate validation, save chkpt and print/plot loss (not too frequently)
  if (i % opt.eval_val_every) == 0 then
    print(os.date())
    -- plot one benchmark sequence
    --       local real_loss = eval_benchmark()
    local real_loss, eval_benchmark_mm= 1,1
    real_losses[i] = real_loss

    plot_real_loss_x, plot_real_loss = getValLossPlot(real_losses)

    -- evaluate validation set
    val_loss = eval_val()
    val_losses[i] = math.max(val_loss, 1e-5)

    train_mm[i] = feval_mm+1
    val_mm[i] = eval_val_mm+1
    real_mm[i] = eval_benchmark_mm+1

    local plot_loss_x, plot_loss = getLossPlot(i, opt.eval_val_every, train_losses)
    local plot_val_loss_x, plot_val_loss = getValLossPlot(val_losses)

    local plot_train_mm_x, plot_train_mm = getValLossPlot(train_mm)
    local plot_val_mm_x, plot_val_mm = getValLossPlot(val_mm)
    local plot_real_mm_x, plot_real_mm = getValLossPlot(real_mm)

    --       print(train_losses)
    --       print(plot_loss)
    local minTrainLoss, minTrainLossIt = torch.min(plot_loss,1)
    local minValidLoss, minValidLossIt = torch.min(plot_val_loss,1)
    local minRealLoss, minRealLossIt = torch.min(plot_real_loss,1)
    minTrainLoss=minTrainLoss:squeeze()
    minTrainLossIt=minTrainLossIt:squeeze()*opt.eval_val_every
    minValidLoss=minValidLoss:squeeze()
    minValidLossIt=minValidLossIt:squeeze()*opt.eval_val_every
    minRealLoss=minRealLoss:squeeze()
    minRealLossIt=minRealLossIt:squeeze()*opt.eval_val_every

    -- TODO there is a bug in best training loss and best training DA
    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','Losses','Training','Valid','Real'))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Current',plot_loss[-1],val_loss,real_loss))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))


    local minTrainLoss, minTrainLossIt = torch.min(plot_train_mm,1)
    local minValidLoss, minValidLossIt = torch.min(plot_val_mm,1)
    local minRealLoss, minRealLossIt = torch.min(plot_real_mm,1)
    minTrainLoss=minTrainLoss:squeeze()      minTrainLossIt=minTrainLossIt:squeeze()*opt.eval_val_every
    minValidLoss=minValidLoss:squeeze()      minValidLossIt=minValidLossIt:squeeze()*opt.eval_val_every
    minRealLoss=minRealLoss:squeeze()        minRealLossIt=minRealLossIt:squeeze()*opt.eval_val_every

    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','MissDA','Training','Valid','Real'))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Current',feval_mm+1,eval_val_mm+1,eval_benchmark_mm+1))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))
    pm('--------------------------------------------------------')

    -- save checkpt
    local savefile = getCheckptFilename(modelName, opt, modelParams)
    saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)


    --       os.execute("th rnnTrackerY.lua -model_name testingY -model_sign r50_l2_n3_m3_d2_b1_v0_li1 -suppress_x 0 -length 5 -seq_name TUD-Crossing")
    -- save lowest val if necessary
    if val_loss <= torch.min(plot_val_loss) then
      pm('*** New min. validation loss found! ***',2)
      local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
      local savefile  = dir .. base .. '_' .. signature .. '_val' .. ext
      saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
    end

    -- save lowest val if necessary
    if real_loss <= torch.min(plot_real_loss) then
      pm('### New min. REAL loss found! ###',2)
      local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
      local savefile  = dir .. base .. '_' .. signature .. '_real' .. ext
      saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
    end

    -- plot
    local lossPlotTab = {}
    table.insert(lossPlotTab, {"Trng loss",plot_loss_x,plot_loss, 'linespoints lt 1'})
    table.insert(lossPlotTab, {"Vald loss",plot_val_loss_x, plot_val_loss, 'linespoints lt 3'})
    --       table.insert(lossPlotTab, {"Real loss",plot_real_loss_x, plot_real_loss, 'linespoints lt 5'})
    table.insert(lossPlotTab, {"Trng MM",plot_train_mm_x, plot_train_mm, 'points lt 1'})
    table.insert(lossPlotTab, {"Vald MM",plot_val_mm_x, plot_val_mm, 'points lt 3'})
    --       table.insert(lossPlotTab, {"Real MM",plot_real_mm_x, plot_real_mm, 'points lt 5'})

    --  local minInd = math.min(1,plot_loss:nElement())
    local maxY = math.max(torch.max(plot_loss), torch.max(plot_val_loss), torch.max(plot_real_loss),
      torch.max(plot_train_mm), torch.max(plot_val_mm), torch.max(plot_real_mm))*2
    local minY = math.min(torch.min(plot_loss), torch.min(plot_val_loss), torch.min(plot_real_loss),
      torch.min(plot_train_mm), torch.min(plot_val_mm), torch.min(plot_real_mm))/2
    local rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",
      opt.eval_val_every-1, i+1, minY, maxY)
    --       rangeStr = string.format("set yrange [%f:%f]", minY, maxY)
    local rawStr = {}
    table.insert(rawStr, rangeStr)
    table.insert(rawStr, 'set logscale y')

    local winTitle = string.format('Loss-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
    plot(lossPlotTab, 2, winTitle, rawStr, 1) -- plot and save (true)
    gnuplot.raw('unset logscale') -- for other plots

    -- Evaluate on MOT15 training
    local meanmets = torch.zeros(1,14)
    if opt.eval_mot15 ~= 0 then
      local timer = torch.Timer()
      local trackerScript = 'rnnTrackerBFSEP'
      --  if stateVel then trackerScript = 'rnnTrackerV' end
      if opt.eval_mot15 < 0 then
        meanmets = runMOT15(trackerScript, modelName, modelSign, valSeqTable)
      else
        meanmets = runMOT15(trackerScript, modelName, modelSign)
      end
      pm('Evaluation took '..timer:time().real..'sec',2)



      local MOTA = meanmets[1][12] -- MOTA is on pos 12
      print('MOTA '..MOTA)
      mot15_mota[i] = MOTA

      --... and plot MOTA
      local plot_mota_x, plot_mota = getValLossPlot(mot15_mota)
      --       sopt = opt -- for evaluation
      local motaPlotTab = {}
      table.insert(motaPlotTab, {"MOTA", plot_mota_x, plot_mota})
      local winTitle = string.format('MOTA-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
      rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",   opt.eval_val_every-1, i+1, -1, 20)

      local rawStr = {}
      table.insert(rawStr, rangeStr)
      plot(motaPlotTab, 4, winTitle, rawStr, 1)

      -- save if lowest mota
      local maxMOTA, maxMOTAIt = torch.max(plot_mota,1)
      maxMOTA = maxMOTA:squeeze()
      maxMOTAIt = maxMOTAIt:squeeze()
      -- local best
      if maxMOTAIt == nil or maxMOTAIt == plot_mota:nElement() then
        pm('*** New max. MOTA found! ***',2)
        local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
        local savefile  = dir .. base .. '_' .. signature .. '_mota' .. ext
        saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)

        -- save MOTA as txt
        csvWrite(string.format('%s/mota_%.1f.txt',outDir,maxMOTA),torch.Tensor(1,1):fill(maxMOTA))  -- write cost matrix
        csvWrite(string.format('%s/bm.txt',outDir,maxMOTA),torch.Tensor(1,1):fill(maxMOTA)) -- write cost matrix

        -- all-time best
        if writeBestMOTA(maxMOTA, modelName, modelSign) then -- update all time best
          savefile = string.format('bin/bestmota-%.1f_x.t7',maxMOTA)
          saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
        end

      else
        pm(string.format('Max MOTA (%.2f) was at iteration %d',maxMOTA,plot_mota_x[maxMOTAIt]),2)
      end


      -------------------------------------------------------
      ------  H U N G A R I A N -----------------------------
      -------------------------------------------------------
      local oc = opt.eval_conf
      pm('Evaluating with Hungarian...',2)
      opt.eval_conf = 'config/XPOSHA.ini'

      local timer = torch.Timer()
      local trackerScript = 'rnnTrackerBFSEP'
      --  if stateVel then trackerScript = 'rnnTrackerV' end
      local meanmetsHA = torch.zeros(1,14)
      if opt.eval_mot15 < 0 then
        meanmetsHA = runMOT15(trackerScript, modelName, modelSign, valSeqTable)
      else
        meanmetsHA = runMOT15(trackerScript, modelName, modelSign)
      end
      pm('Evaluation took '..timer:time().real..'sec',2)



      local MOTAHA = meanmetsHA[1][12] -- MOTA is on pos 12
      print('MOTA Hungarian'..MOTAHA)
      mot15_motaHA[i] = MOTAHA

      --... and plot MOTA
      local plot_mota_xHA, plot_motaHA = getValLossPlot(mot15_motaHA)
      --       sopt = opt -- for evaluation
      local motaPlotTabHA = {}
      table.insert(motaPlotTabHA, {"MOTA", plot_mota_xHA, plot_motaHA})
      local winTitle = string.format('MOTA-HA-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
      rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",   opt.eval_val_every-1, i+1, -1, 20)

      local rawStr = {}
      table.insert(rawStr, rangeStr)
      plot(motaPlotTabHA, 4, winTitle, rawStr, 1)

      -- save if lowest mota
      local maxMOTAHA, maxMOTAItHA = torch.max(plot_motaHA,1)
      maxMOTAHA = maxMOTAHA:squeeze()
      maxMOTAItHA = maxMOTAItHA:squeeze()
      -- local best
      if maxMOTAItHA == nil or maxMOTAItHA == plot_motaHA:nElement() then
        pm('*** New max. MOTA HA found! ***',2)
        local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
        local savefile  = dir .. base .. '_' .. signature .. '_mota' .. ext
        --    saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)

        -- save MOTA as txt
        csvWrite(string.format('%s/motaHA_%.1f.txt',outDir,maxMOTAHA),torch.Tensor(1,1):fill(maxMOTAHA))  -- write cost matrix
        csvWrite(string.format('%s/bmHA.txt',outDir,maxMOTAHA),torch.Tensor(1,1):fill(maxMOTAHA)) -- write cost matrix

        -- all-time best
        if writeBestMOTA(maxMOTAHA, modelName, modelSign,'bestmotaHA.txt') then -- update all time best
          savefile = string.format('bin/bestmotaHA-%.1f_x.t7',maxMOTA)
          saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
        end

      else
        pm(string.format('Max MOTA HA (%.2f) was at iteration %d',maxMOTAHA,plot_mota_xHA[maxMOTAItHA]),2)
      end
      opt.eval_conf = oc
      ----------------------------------------------------------


    end -- if eval_mot15


    printModelOptions(opt) -- print parameters
  end

  if i == 1 or i % (opt.print_every*10) == 0 then
    printModelOptions(opt)
    printTrainingHeadline()
  end -- headline

end

print(string.format('%20s%10.2f%7s','total time meas.',mtime,''))
print(string.format('%20s%10.2f%7s','total time',ggtime:time().real,''))
