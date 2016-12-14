------------------------------------------------------
-- Online Data Association using RNNs
-- A. Milan, S. H. Rezatofighi, K. Schinlder, I. Reid
-- arxiv 2015
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
-- require 'runMOT15'

require 'util.misc'  -- clone_list
require 'auxDA'


torch.manualSeed(1)

nngraph.setDebug(true)



local model_utils = require 'util.model'
--local LSTM = require 'model.LSTMX'
-- local RNN = require 'model.RNNBFUPDONLY'
-- local RNN = require 'model.RNNBFPREDONLY'
local RNN = require 'model.RNNDA'
-- local RNN = require 'model.RNNBFPREDUPD'
-- local RNN = require 'model.RNNBFPREDDA'
-- local RNN = require 'model.RNNBF'
--local GRU = require 'model.GRUX'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a simple trajectory model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-config', './config/configDA.txt', 'config file')
cmd:option('-rnn_size', 10, 'size of RNN internal state')
cmd:option('-num_layers',2,'number of layers in the RNN / LSTM')
cmd:option('-model_index',1,'1=lstm, 2=gru, 3=rnn')
cmd:option('-temp_win',10, 'temporal window history')
cmd:option('-batch_size',1,'number of frames to consider (1=online)')
cmd:option('-max_n',3,'Max number of targets per frame')
cmd:option('-max_m',3,'Max number of measurements per frame')
cmd:option('-max_nf',3,'Max number of false targets per frame')
cmd:option('-state_dim',1,'state dimension (1-4)')
cmd:option('-loss_type',1,'1=BCE, 2=MLM, 3=MLSM, 4=MSE, 5=KL, 6=ABSERR, 7=NLL')
cmd:option('-order_dets',1,'order detections (HA) before feeding')
cmd:option('-kappa',5,'gt loss weighting')
cmd:option('-lambda',5,'pred loss weighting')
cmd:option('-mu',1,'label loss weighting')
cmd:option('-nu',1,'ex loss weighting')
cmd:option('-xi',1,'smoothness termination')
cmd:option('-vel',0,'include velocity into state')
cmd:option('-linp',1,'use labels inputs')
cmd:option('-einp',0,'use labels inputs')
cmd:option('-one_hot',1,'use one hot encoding')
cmd:option('-bce',1,'use BCE instead of CNLL loss')
cmd:option('-pwd_mode',0,'0 = squeezed dist, 1 = abs. dist per dim, 2 = overlap (4 dim only)')
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
cmd:option('-use_gt_input', 1, 'use gt as current input at t')
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
cmd:option('-predupd_model','0225Al-1','prediction update module')
cmd:option('-predupd_model_sign','r300_l1_n5_m5_d4_b1_v0_li0','predupd signature')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()


-- parse input params
opt = cmd:parse(arg)
-- overwrite params with config if given
if string.len(opt.config) > 0 then opt = parseConfig(opt.config, opt) end

if opt.real_data==1 and opt.state_dim<2 then error('ERROR: Real Data requires dim >= 2') end
if opt.real_data==0 then opt.real_dets = 0 end

-- we need at least N detections slots
if opt.max_m < opt.max_n then opt.max_m = opt.max_n + 0 end

-- set number of false tracks manually
opt.max_nf = opt.max_m - opt.max_n
opt.max_nf = 0; print('!!!!!')
-- if opt.max_m > opt.max_n then opt.max_m = opt.max_n + 0 end
-- opt.max_m = opt.max_n print('MAKE SURE maxDets=maxTargets!!!') sleep(.1)



if opt.real_dets ~= 0 then opt.det_noise, opt.det_fail, opt.det_false = 0,0,0 end

stateVel = opt.vel~= 0
-- if stateVel then LSTM = require 'model.LSTMBEV' error('no vel support') end
-- exVar = opt.exvar ~= 0
updLoss = opt.kappa ~= 0
predLoss = opt.lambda ~= 0
daLoss = opt.mu ~= 0
exVar = opt.nu ~= 0
smoothVar = opt.xi ~= 0
if opt.pwd_mode == 2 and opt.state_dim ~= 4 then opt.pwd_mode = 0 end

fullPWD = opt.pwd_mode == 1  -- N * M

--   fullPWD = true  -- N * M * D

-- if exVar then error('not implemented exvar') end

val_temp_win = opt.temp_win

opt.model = 'lstm'
if opt.model_index==2 then opt.model='gru'
elseif opt.model_index==3 then opt.model='rnn'
end

-- Profiler
local ProFi = nil
if opt.profiler > 0 then ProFi = require 'util.ProFi' end
if ProFi then ProFi:start(); ProFi:setInspect('feval',2) end

profTable = {}
ggtime = torch.Timer()

checkCuda() 			-- check for cuda availability
torch.manualSeed(opt.seed)	-- manual seed for deterministic runs

miniBatchSize = opt.mini_batch_size

if opt.lrng_rate_decay_after < 0 then opt.lrng_rate_decay_after = opt.max_epochs / 10 end
-- Where do we find the state location and existance in the LSTM output?

local nHiddenInputs = 1
if opt.model=='lstm' then nHiddenInputs = 2 end

opt.statePredIndex = opt.num_layers*nHiddenInputs
if updLoss then opt.statePredIndex = opt.num_layers*nHiddenInputs+1 end
-- if daLoss then opt.statePredIndex = opt.num_layers+2 end

opt.statePredIndex2 = opt.statePredIndex+1

if daLoss then opt.daPredIndex = opt.statePredIndex+1 end
if predLoss then  opt.daPredIndex = opt.statePredIndex2+1 end		-- second state output
-- if exVar then opt.exPredIndex = opt.daPredIndex+1 end		-- binary indicator
opt.exPredIndex = opt.daPredIndex+1
opt.daSumPredIndex = opt.daPredIndex+1
opt.daSum2PredIndex = opt.daPredIndex+2
opt.exSmoothPredIndex = opt.daSum2PredIndex+1



print('state index  '..opt.statePredIndex)
print('state index2 '..opt.statePredIndex2)
if daLoss then print('da index     '..opt.daPredIndex) end
if daLoss and opt.bce~=0 then
  print('da sum index     '..opt.daSumPredIndex)
  print('da sum2 index     '..opt.daSum2PredIndex)
end
if exVar then print('ex index     '..opt.exPredIndex) end
if smoothVar then print('SmEx index     '..opt.exSmoothPredIndex) end

-- abort()
-- opt.save_plots = opt.save_plots ~= 0
-- print(opt.save_plots)

local colorDetsShuffled = true 		-- colors correspond to actual row
-- local colorDetsShuffled = false
------------------------------
----- BUILDING THE MODEL -----
------------------------------
modelName = 'default'		-- base name
if string.len(opt.config)>0 then
  local fp,fn,fe = fileparts(opt.config); modelName = fn
end
-- a list of model-specific parameters... These cannot change across models
modelParams = {'model_index','rnn_size', 'num_layers','max_n','max_m','state_dim','batch_size', 'vel', 'linp'}
dataParams={'synth_training','synth_valid','mini_batch_size',
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
  protos.rnn = RNN.rnn(opt)
  --   if opt.model == 'lstm' then protos.rnn = LSTM.lstm(opt)
  --   elseif opt.model == 'gru' then protos.rnn = GRU.gru(opt)
  --   elseif opt.model == 'rnn' then protos.rnn = RNN.rnn(opt)
  --   else print('Unknown model. Take LSTM'); protos.rnn = LSTM.lstm(opt) end
  --   local msec = nn.MSECriterion()
  --   protos.criterion = nn.ParallelCriterion()
  --   protos.criterion:add(msec, opt.gt_loss)
  --   protos.criterion:add(msec, opt.det_loss)
  --   protos.criterion = nn.MSECriterion()
  local lambda = opt.lambda
  local nllc = nn.ClassNLLCriterion()
  local bce = nn.BCECriterion()
  local mse = nn.MSECriterion()
  local abserr = nn.AbsCriterion()
  local kl = nn.DistKLDivCriterion()
  local mlm = nn.MultiLabelMarginCriterion()
  local mlsm = nn.MultiLabelSoftMarginCriterion()
  protos.criterion = nn.ParallelCriterion()
  if updLoss then protos.criterion:add(nn.MSECriterion(), opt.kappa) end
  if predLoss then protos.criterion:add(nn.MSECriterion(), opt.lambda) end
  if daLoss then
    if opt.bce==0 then
      --       1=BCE, 2=MLM, 3=MLSM, 4=MSE, 5=KL, 6=ABSERR, 7=NLL
      --       protos.criterion:add(nllc, opt.mu)
      --       protos.criterion:add(kl, opt.mu)
      --       protos.criterion:add(mlsm, opt.mu)
      --       protos.criterion:add(mse, opt.mu)
      --       protos.criterion:add(abserr, opt.mu)
      --       protos.criterion:add(mlm, opt.mu)
      --       protos.criterion:add(mlsm, opt.mu)
      if opt.loss_type == 1 then
        protos.criterion:add(bce, opt.mu)
      elseif opt.loss_type == 2 then
        protos.criterion:add(mlm, opt.mu)
      elseif opt.loss_type == 3 then
        protos.criterion:add(mlsm, opt.mu)
      elseif opt.loss_type == 4 then
        protos.criterion:add(mse, opt.mu)
      elseif opt.loss_type == 5 then
        protos.criterion:add(kl, opt.mu)
      elseif opt.loss_type == 6 then
        protos.criterion:add(abserr, opt.mu)
      elseif opt.loss_type == 7 then
        protos.criterion:add(nllc, opt.mu)
      end


    else
      --       protos.criterion:add(nllc, opt.mu)
      protos.criterion:add(mlm, opt.mu)
      --      protos.criterion:add(mlsm, opt.mu)
      --       protos.criterion:add(kl, opt.mu)
      --       protos.criterion:add(mlsm, opt.mu)
      protos.criterion:add(bce, opt.bce) -- == 1
      protos.criterion:add(bce, opt.bce) -- <= 1
    end
  end

  if exVar then
    protos.criterion:add(bce, opt.nu)
    --     protos.criterion:add(nn.MSECriterion(), opt.nu)
  end
  if smoothVar then protos.criterion:add(nn.MSECriterion(), opt.xi) end

  --   print(protos)

end

logifyDA = true
-- logifyDA = false

-------------
-- GLOBALS --
stateDim = opt.state_dim
fullStateDim = stateDim if stateVel then fullStateDim = stateDim * 2 end
print("State dimension: "..stateDim)
print("Full State dimension: "..fullStateDim)
maxDetThr = opt.det_thr

-- size of data tensors to work with
maxRTargets, maxDets = opt.max_n, opt.max_m
maxFTargets = opt.max_nf
maxTargets = maxRTargets+maxFTargets
nClasses = maxDets+1
xSize = stateDim*maxTargets
dSize = stateDim*maxDets
fullxSize = fullStateDim*maxTargets


-- print and log options and setting
local scriptFileName = 'trainDA.lua'
-- if stateVel then scriptFileName = 'trainV.lua' end

--print()
--abort()
-- printOptions(opt)
local _,_,_,modelSign = getCheckptFilename(modelName, opt, modelParams)
local outDir = string.format('%s/tmp/%s_%s/',getRNNTrackerRoot(), modelName, modelSign)
if not lfs.attributes(outDir, 'mode') then lfs.mkdir(outDir) end
local ofile = assert(io.open( outDir .. 'opt.txt', "w"))
local cfile = assert(io.open( outDir .. 'cmd.txt', "w"))
local cffile = assert(io.open( outDir .. 'cmdfull.txt', "w"))
cffile:write(string.format('th %s ',scriptFileName))
cfile:write(string.format('th %s ',scriptFileName))
for k,v in pairs(opt) do
  ofile:write(string.format("%22s  %s\n",k,tostring(v)))	-- all options as a table
  if string.len(tostring(v)) > 0 then
    cffile:write(string.format('-%s %s ',k,tostring(v)))	-- all options as command line input
  end
end
for k, v in pairs(arg) do
  if k > 0 then
    local hyphen = ''; if k%2==1 then hypthen = '' end
    cfile:write(string.format('%s%s ',hyphen,tostring(v)))	-- passed options as command line input
  end
end
ofile:close()
cfile:close()
cffile:close()



-- copy this file as a backup
cpStr = string.format('cp %s %s',scriptFileName, outDir)
os.execute(cpStr)

-- cp XPOS to tmp and append line
cpStr = string.format('cp ./matlab/config/%s %s',opt.eval_conf, outDir)
os.execute(cpStr)
newEvalConf = outDir..opt.eval_conf
print(newEvalConf)
local cfile = assert(io.open(newEvalConf, "a"))
print('writing '..string.format('da_model = %s_%s',modelName,modelSign))
cfile:write(string.format('da_model \t= %s_%s',modelName,modelSign))
cfile:close()
opt.eval_conf = '../'..newEvalConf
-- abort()


-- the initial state of the cell/hidden states
init_state = getInitState(opt, miniBatchSize)

-- ship the model to the GPU if desired
for k,v in pairs(protos) do v = dataToGPU(v) end


-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- opt.rand_par_rng = 0.08 -- Karpathy's
-- opt.rand_par_rng = 0.1


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

if lfs.attributes('/home/amilan/','mode') then
  print('Drawing Graph')
  graph.dot(protos.rnn.fg, 'RNNBFF', 'RNNBFForwardGraph_DA')
  graph.dot(protos.rnn.bg, 'RNNBFB', 'RNNBFBackwardGraph_DA')
end
-- abort('graph')



local trTracksTab, trDetsTab, trOrigDetsTab = {}, {}, {}

local valTracksTab, valDetsTab, valOrigDetsTab = {}, {}, {}
local realData = opt.real_data == 0
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

print('Training Data...')
print(trSeqTable)

print('Validation Data...')
print(valSeqTable)

print(string.format('Real Trajectories: %d\tReal Detections: %d',opt.real_data, opt.real_dets))

pm(string.format('reading data...'),2)
trTracksTab, trDetsTab = getTracksAndDetsTables(trSeqTable, maxTargets, maxDets, nil, false)
pm(string.format('   ...done'),2)

-- adjust sample size
opt.synth_training = math.max(miniBatchSize,opt.synth_training)
opt.synth_valid = math.max(miniBatchSize,opt.synth_valid)

-- adjust sample size
opt.synth_training = math.floor(opt.synth_training/miniBatchSize) * miniBatchSize
opt.synth_valid = math.floor(opt.synth_valid/miniBatchSize) * miniBatchSize



-- do we want to fix N?
local fixed_n = opt.fixed_n ~= 0

imSizes = getImSizes(trSeqTable)
imSizes = getImSizes(valSeqTable, imSizes)


-- GET DATA ---
LOAD = true
trTracksTab, trDetsTab, trLabTab, trExTab, trSeqNames = prepareData('train', trSeqTable, trSeqTable, false)
local o=opt.temp_win opt.temp_win=val_temp_win
valTracksTab, valDetsTab, valLabTab, valExTab, valSeqNames = prepareData('validation', valSeqTable, trSeqTable,  false)
-- realTracksTab, realDetsTab, realLabTab, realExTab, realSeqNames = prepareData('real', valSeqTable, trSeqTable,  true)
opt.temp_win=o
for k,v in pairs(trTracksTab) do
  local N,F,D = getDataSize(v)
  --   trTracksTab[k] = torch.rand(N,F,D):float() - 0.5
  --   trTracksTab[k] = v * (torch.rand(N,F,D)*opt.det_noise-0.5):float()
end

for k,v in pairs(trDetsTab) do
  local N,F,D = getDataSize(v)
  --   trDetsTab[k] = v + 0.2
  --   trDetsTab[k] = v + torch.rand(1):squeeze()
  --   trTracksTab[k] = v * (torch.rand(N,F,D)*opt.det_noise-0.5):float()
end

-- for k,v in pairs(trDetsTab) do
--   for l,m in pairs(trDetsTab[k]) do
--     trDetsTab[k][l] = 0
--   end
-- end

-- for k,v in pairs(valDetsTab) do valDetsTab[k] = 0 end



-- printAll(trTracksTab[1], trDetsTab[1], trLabTab[1])
-- printAll(realTracksTab[1], realDetsTab[1], realLabTab[1])
-- abort()
print('Training batches:   '..tabLen(trTracksTab))
print('Validation batches: '..tabLen(valTracksTab))

function showData(trackTable, detTable, seqNamesTable, n)

  n = n or tabLen(trackTable)
  local maxShow = math.min(20,n)
  for seq=1,maxShow do
    local tracks = trackTable[seq]
    local detections = detTable[seq]
    local seqName = seqNamesTable[seq]

    local N,F,D=getDataSize(detections:sub(1,maxDets))
    local da = torch.IntTensor(N,F)
    if colorDetsShuffled then
      for i=1,maxDets do da[i]=(torch.ones(1,opt.temp_win)*i) end -- color dets shuffled
    end

    plotTab = getTrackPlotTab(tracks:sub(1,maxTargets):float(), {}, 1)
    plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, da)

    plot(plotTab, winID, string.format('%s-%s-%06d','Traning-Data',seqName[1], seq), nil, 1) -- do not save first
    --     sleep(.2)
  end

end

-- showData(trTracksTab, trDetsTab, trSeqNames)


---------------------
ffOne = torch.IntTensor(1,maxTargets*2):fill(1)
firstFrameEx = torch.IntTensor(1,maxTargets*2):fill(1)
for m=1,miniBatchSize do firstFrameEx=firstFrameEx:cat(ffOne,1) end
firstFrameEx=firstFrameEx:sub(2,-1)

firstFrameEx = dataToGPU(firstFrameEx)
firstFrameExT = dataToGPU(ffOne)


function getGTState(t)
  local loctimer=torch.Timer()
  local locTracks = tracks
  local locVels = vels
  --   print('aa', miniBatchSize)
  --   print(tracks)

  --   local outLocs = torch.zeros(miniBatchSize,fullxSize):float()
  local outLocs = {}

  if stateVel then
    outLocs = torch.zeros(miniBatchSize,fullxSize):float()
    local outtr = tracks[{{},{t+1}}]:reshape(miniBatchSize,xSize) -- output should be next location
    local outvel = vels[{{},{t+1}}]:reshape(miniBatchSize,xSize) -- output should be next location

    -- TODO make efficient
    for m=1,miniBatchSize do
      for s=1,fullxSize,2 do
        outLocs[m][s] = outtr[m][math.ceil(s/2)]
        outLocs[m][s+1] = outvel[m][math.ceil(s/2)]
      end
    end
  else
    --     print(tracks)
    --     print(miniBatchSize, fullxSize)
    --   local GTWeight = 0.5
    --   outLocs = (detections[{{},{t+1}}]*(1-GTWeight) + locTracks[{{},{t+1}}]*(GTWeight)):reshape(miniBatchSize,fullxSize)

    outLocs = tracks[{{},{t+1}}]:reshape(miniBatchSize,fullxSize)

  end
  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return outLocs
end

labOffset = torch.zeros(opt.mini_batch_size*maxTargets)
for mb=2,miniBatchSize do
  local mbStartT = maxTargets * (mb-1)+1
  local mbEndT =   maxTargets * mb

  labOffset[{{mbStartT,mbEndT}}] = (mb-1) * opt.max_m

end
labOffset = dataToGPU(labOffset)

-- labOffsetT = dataToGPU()

function getGTDet(t, labels)
  local loctimer=torch.Timer()
  local locTracks = tracks:clone()
  local locDets  = detections:clone()

  local outLocs = {}



  if stateVel then
    error('aehm... velocity for detections...')
  else

    local frameLabs = labels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
    if miniBatchSize>1 then frameLabs = (frameLabs+labOffset) end

    if opt.max_m>1 then
      outLocs = locDets[{{},{t+1}}]:index(1,frameLabs:long())
    else
      outLocs = locDets[{{},{t+1}}]
    end

    -- replace missing detections with GT locations
    locTracks = locTracks:narrow(2,t+1,1)
    outLocs[outLocs:eq(0)] = locTracks[outLocs:eq(0)]
    outLocs = outLocs:reshape(miniBatchSize, xSize)
    --
  end

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end




  return outLocs
    --   return getGTState(t)
end


-- get ground truth data assoication
function getGTDA(t)
  local loctimer=torch.Timer()

  --   print(labels)
  --   print(miniBatchSize, maxTargets)

  --   1=BCE, 2=MLM, 3=MLSM, 4=MSE, 5=KL, 6=ABSERR, 7=NLL

  local DA = nil
  if opt.loss_type == 1 then
    DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize,maxTargets*nClasses)
  elseif opt.loss_type == 2 then
    --   if opt.bce==0 then
    DA = labels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
    -----------------------------
    -- MultiLabelMarginCriterion
    local multilabTensor = torch.zeros(miniBatchSize, maxTargets*nClasses)
    for mb = 1,opt.mini_batch_size do
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb

      for tar = 1,maxTargets do
        multilabTensor[mb][tar] = DA[mbStart+tar-1] + (tar-1)*nClasses
      end
    end
    DA = multilabTensor:reshape(miniBatchSize,maxTargets*nClasses)
    -----------------------------
  elseif opt.loss_type == 3 then
    DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize,maxTargets*nClasses)
  elseif opt.loss_type == 4 then
    DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize,maxTargets*nClasses)
  elseif opt.loss_type == 5 then
    DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize,maxTargets*nClasses)
  elseif opt.loss_type == 6 then
    DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize,maxTargets*nClasses)
  elseif opt.loss_type == 7 then
    --   if opt.bce==0 then
    DA = labels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
    -----------------------------
    -- MultiLabelMarginCriterion
--    local multilabTensor = torch.zeros(miniBatchSize, maxTargets*nClasses)
--    for mb = 1,opt.mini_batch_size do
--      local mbStart = opt.max_n * (mb-1)+1
--      local mbEnd =   opt.max_n * mb
--
--      for tar = 1,maxTargets do
--        multilabTensor[mb][tar] = DA[mbStart+tar-1] + (tar-1)*nClasses
--      end
--    end
--    DA = multilabTensor:reshape(miniBatchSize,maxTargets*nClasses)
  end
  --     DA:fill(0)
  --     DA[1][1]=1
  --     print(labels[{{},{t+1}}])

  --     print(DA)
  --     sleep(1)
  --     abort()
  --   else
  --     DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize*maxTargets*nClasses)
  --     DA = labels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
  --   end
  --   print(t)
  --   print(labels)
  --   print(DA)
  --   print(DA:reshape(miniBatchSize*maxTargets*nClasses))
  --   abort()

  --   DA = DA:reshape(miniBatchSize,
  --   DA = DA:reshape(miniBatchSize*maxTargets,nClasses)

  DA = dataToGPU(DA)
--  print(labels)
--  abort()

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return DA
end

function getPredDet(t, predDA)
  local loctimer=torch.Timer()
  local locTracks = tracks:clone()
  local locDets  = detections:clone()

  local outLocs = {}

  local labFromPred = torch.zeros(labels:size()):float()
  --   print(predDA)
  --   abort()
  --   print(maxTargets, miniBatchSize, maxDets)

  --   local ll = getLabelsFromLL(predDA:reshape(maxTargets*miniBatchSize,1,maxDets))
  local mv,mi=torch.max(predDA,2)
  labFromPred[{{},{t+1}}] = mi
  --   labFromPred = labels:clone()
  --   print(labFromPred)

  --   print(torch.max(predDA,2))
  --   abort()


  local frameLabs = labFromPred[{{},{t+1}}]:reshape(miniBatchSize*maxTargets)
  --   print(frameLabs)
  --   print(labOffset)
  if miniBatchSize>1 then frameLabs = (frameLabs+labOffset) end

  outLocs = locDets[{{},{t+1}}]:index(1,frameLabs:long())

  -- replace missing detections with GT locations
  locTracks = locTracks:narrow(2,t+1,1)
  outLocs[outLocs:eq(0)] = locTracks[outLocs:eq(0)]

  local GTWeight = 0.5
  outLocs = outLocs*(1-GTWeight) + locTracks*(GTWeight)
  --   print(labFromPred)
  --   print(predDA)
  --
  --   printDim(detections)
  --   printDim(tracks)
  --   printDim(outLocs)
  --
  --   abort()

  outLocs = outLocs:reshape(miniBatchSize, xSize)


  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return outLocs
end

function getPredDetW(t, predDA)
  local loctimer=torch.Timer()
  local locTracks = tracks:clone()
  local locDets  = detections:clone()

  local outLocs = {}

  --   print(predDA)
  local probDA = torch.exp(predDA)
  --   print(probDA)
  local Z = torch.sum(probDA,2)
  --   print(Z)
  for i=1,probDA:size(1) do
    probDA[i] = probDA[i]/Z[i][1]
  end
  --   print(probDA)
  local Z = torch.sum(probDA,2)

  outLocs = zeroTensor3(maxTargets, 1, stateDim)
  for i=1,maxTargets do
    for d=1,maxDets do
      local targetPos = locDets[d][t+1]
      if targetPos[1] == 0 then targetPos = locTracks[d][t+1] end

      --       print(outLocs[i])
      --       print(probDA[i][d])
      --       print(targetPos)
      outLocs[i] = outLocs[i] + targetPos*probDA[i][d]
    end
  end

  outLocs = outLocs:reshape(miniBatchSize, xSize)


  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return outLocs
end


function getGTEx(t)
  local loctimer=torch.Timer()
  -- which targets exist?
  --   local nextFrame = tracks[{{},{t+1},{1}}]:reshape(miniBatchSize*maxTargets) -- first dim
  -- --   local exTar = torch.IntTensor(miniBatchSize*maxTargets):fill(2) -- default, does not exist
  --   local exTar = zeroTensor1(miniBatchSize*maxTargets) -- default, does not exist
  -- --   local exTar = zeroTensor1(miniBatchSize*maxTargets) -- default, does not exist
  --   exTar[nextFrame:ne(0)] = 1 -- these ones exist

  exTar = exlabels[{{},{t+1}}]:reshape(miniBatchSize*maxTargets):float()

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return exTar
end

function getGTDAsum(t)
  local loctimer=torch.Timer()


  -- row sums (== constraint)
  local DAsum = torch.ones(miniBatchSize*maxTargets)
  DAsum = dataToGPU(DAsum)


  -- column sums  (<= constraint)
  local DA = getOneHotLab(labels[{{},{t+1}}], miniBatchSize==1):reshape(miniBatchSize*maxTargets,nClasses)
  --   print(labels[{{},{t+1}}])
  --   print(DA)
  --   abort()
  DA=DA:narrow(2,1,maxTargets)

  local DAsum2 = torch.zeros(1,maxTargets):float()
  for mb=1,miniBatchSize do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb

    DAsum2 = DAsum2:cat(torch.sum(DA[{{mbStart, mbEnd},{}}], 1):reshape(1,maxTargets),1)
  end
  DAsum2=DAsum2:sub(2,-1):reshape(miniBatchSize*maxTargets,1)
  --   print(DAsum2)
  --   abort()

  DAsum2 = dataToGPU(DAsum2)


  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return DAsum, DAsum2
end

function getInfeasibleSolution()



  local possibleAssignments = math.pow(2,maxTargets*nClasses)

  local infSol = {}

  for s=1,possibleAssignments do
    local binCode = torch.Tensor(toBits(s, maxTargets*nClasses))
    ass = binCode:reshape(maxTargets,nClasses)
    --   print(ass)
    local feasible = true
    local sumAllEntries = torch.sum(binCode)
    local sumOverColumns = torch.sum(ass,2) -- ==
    local sumOverRows = torch.sum(ass:narrow(2,1,maxTargets),1) -- <=
    local allOnes = torch.ones(maxTargets, 1)
    --   print(sumOverColumns:ne(1))
    --   print(torch.sum(sumOverRows:gt(1)))

    --   print(ass)
    if sumAllEntries ~= maxTargets then
      feasible = false
      --     print('incorrect sum of assignments maxTar assigments '..sumAllEntries)
    elseif torch.sum(sumOverColumns:ne(1)) > 0 then
      feasible = false
      --     print('incorrect column sum '..torch.sum(sumOverColumns:ne(1)))
    elseif torch.sum(sumOverRows:gt(1)) > 0 then
      feasible = false
      --     print('incorrect row sum '..torch.sum(sumOverRows:gt(1)))
    end

    --   print(feasible)
    --     if feasible then print(ass) end
    if not feasible then
      table.insert(infSol, ass)
    end

  end

  local retInfSol = torch.zeros(1, nClasses)
  for mb = 1,miniBatchSize do
    local randSol = math.random(tabLen(infSol))
    retInfSol = retInfSol:cat(infSol[randSol]:reshape(maxTargets,nClasses),1)
  end
  retInfSol = retInfSol:sub(2,-1)

  retInfSol = dataToGPU(retInfSol)
  return retInfSol


end





function plotProgress(state, detections, predTracks, predDA, predEx, winID, winTitle,predTracks2)
  local loctimer=torch.Timer()
  local loc = state:clone()
  if globiter == nil then globiter = 0 end
  if itOffset == nil then itOffset = 0 end
  predDA = predDA:sub(1,maxTargets):float()

  --normalize for plotting
  if opt.bce==1 then
    for tar=1,maxTargets do
      for t=1,predDA:size(2) do
        local probLine = predDA[{{tar},{t}}]:reshape(nClasses)
        probLine = probLine / torch.sum(torch.exp(probLine))
        predDA[{{tar},{t}}] = probLine
      end
    end
    --       predDA = torch.log(predDA)
  end

  local predExO=predEx:clone()
  if exVar then predEx = predEx:sub(1,maxTargets):float() else predEx = nil end

  --     print(predDA)
  --     print(predExBin)
  --     abort()
  --     print(predDA)
  local DA = getLabelsFromLL(predDA, false)
  --     print(DA)
  --     abort()
  --     print(labels)
  plotTab={}
  --     local da = associateDetections(detections:sub(1,maxDets), tracks:sub(1,maxTargets))
  local N,F,D=getDataSize(detections:sub(1,maxDets))
  local da = torch.IntTensor(N,F)
  --     print(detections:sub(1,maxDets):narrow(3,1,1):squeeze(), tracks:sub(1,maxDets):narrow(3,1,1):squeeze(),
  -- 	  predTracks:sub(1,maxDets):narrow(3,1,1):squeeze(), predDA:sub(1,maxDets))
  if colorDetsShuffled then
    for i=1,maxDets do da[i]=(torch.ones(1,opt.temp_win)*i) end -- color dets shuffled
  end
  fullDA = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), DA, 2)
  --     print(da)
  --     print(fullDA)
  --   printDim(detections:sub(1,maxDets))
  --   abort()
  --     print(predTracks)
  plotTab = getTrackPlotTab(tracks:sub(1,maxTargets):float(), plotTab, 1)
  plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, da)
  --     plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, fullDA,0.1)
  if daLoss then
    -- TODO REMOVE GLOBAL STATE
    GTSTATE = state:clone()
    --       print(GTSTATE)
    --       plotTab = getDAPlotTab(predTracks:sub(1,maxTargets):float(), detections:sub(1,maxDets):float(), plotTab, fullDA, predEx, 0, predDA)
    plotTab = getDAPlotTab(state:sub(1,maxTargets):narrow(2,2,opt.temp_win-1):float(), detections:sub(1,maxDets):float(), plotTab, fullDA, predEx, 0, predDA)
  end
  plotTab = getTrackPlotTab(predTracks:sub(1,maxTargets):float(), plotTab, 2, nil, predEx, 1)
  plotTab = getTrackPlotTab(predTracks2:sub(1,maxTargets):float(), plotTab, 3, nil, predEx, 1, nil)

  plotTab = getExPlotTab(plotTab, predExO, 1)
  --     print(plotTab)
  --     abort()
  plot(plotTab, winID, string.format('%s-%06d',winTitle,globiter+itOffset), nil, opt.save_plots) -- do not save first
  --     print(plotTab[7][2])
  --     abort()
  sleep(.1)
  --     sleep(2)
  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
end



function getPredAndGTTables(predState, predState2, predDA, predDAsum, predDAsum2, predEx, t, smoothnessEx)
  local loctimer=torch.Timer()


  --     GTDA[t]=DA

  local input, output = {}, {}

  -- LOC FIRST
  local GTLocs = {}
  if updLoss or predLoss then
    local GTLocs = getGTState(t)
  end

--   if updLoss then


    --     local detPred = getPredDet(t, predDA)
    --     local detPred = getGTDet(t, labels)
--     table.insert(input, predState)	-- predicted (updated) state
--     table.insert(output, GTLocs)	-- detection at predicted DA
--   end



  --     print(output)

--   if predLoss then
    --       local GTLocs = getGTState(t)


    --       local detGT = getGTDet(t, labels)
    --       local labFromPred = torch.zeros(labels:size())
    --       local ll = getLabelsFromLL(predDA:reshape(maxTargets*miniBatchSize,1,maxDets))
    --       labFromPred[{{},{t+1}}] = ll
    --       local detGT = getGTDet(t, labFromPred)
--     table.insert(input, predState2)
    --       print(detPred)
    --       abort()
--     table.insert(output, GTLocs)
--   end

  if daLoss then
    local GTDA = getGTDA(t)
    table.insert(input, predDA)
    table.insert(output, GTDA)

    if opt.bce~=0 then
      local GTDASum, GTDASum2 = getGTDAsum(t)

      -- 	local infSol = getInfeasibleSolution()
      -- 	table.insert(input, predDAsum)
      -- 	table.insert(output, infSol)

      table.insert(input, predDAsum)
      table.insert(output, GTDASum)

      table.insert(input, predDAsum2)
      table.insert(output, GTDASum2)
    end
  end

  if exVar then
    local exTar = getGTEx(t)
    table.insert(input, predEx)
    table.insert(output, exTar)

  end

  -- SMOOTHNESS
  if smoothVar then
    table.insert(input, smoothnessEx)
    table.insert(output, torch.zeros(maxTargets):float())

  end




  --     print(output)
  --     abort()
  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return input, output
end


function eval_val()
  local tmpOpt=deepcopy(opt) -- turn it off for testing
  --   opt.use_gt_input=0
  --   opt.mini_batch_size = 1
  --   miniBatchSize = 1
  local sameWinSize = val_temp_win == opt.temp_win
  if not sameWinSize then
    opt.temp_win=val_temp_win
  end

  local tL = tabLen(valTracksTab)
  local loss = 0
  local T = opt.temp_win - opt.batch_size
  --   print(tL)
  local plotSeq = math.random(tL)
  TRAINING = false
  --   TRAINING = true
  for seq=1,tL do


    if not sameWinSize then
      protosrnn = protos.rnn:clone()
      protoscrit = protos.criterion:clone()
      protosrnn:evaluate()
    end


    tracks = valTracksTab[seq]:clone()
    detections = valDetsTab[seq]:clone()
    labels = valLabTab[seq]:clone()
    exlabels = valExTab[seq]:clone()
    labels:fill(1)
    exlabels:fill(0)

    --     detections, labels = reshuffleDetsAndLabels2(detections, labels)
    if stateVel then vels = valVelTab[seq]:clone() end

    ----- FORWARD ----
    local initStateGlobal = clone_list(init_state)
    local rnn_state = {[0] = initStateGlobal}
    --   local predictions = {[0] = {[opt.statePredIndex] = detections[{{},{t}}]}}
    local predictions = {}
    local stateLocs, stateLocs2, stateDA, stateEx, DAsum, DAsum2 = {}, {}, {}, {}, {}, {}
    local smoothnessEx = {}
    local T = opt.temp_win - opt.batch_size
    for t=1,T do

      if opt.freq_reshuffle~= 0 then
        detections, labels = reshuffleDetsAndLabels2(detections, labels, t)
      end

      local lst={}

      local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)		-- get combined RNN input table



      if not sameWinSize then
        lst = protosrnn:forward(rnninp)
        predictions[t] = {}
        for k,v in pairs(lst) do predictions[t][k] = v:clone() end -- deep copy
      else
        clones.rnn[t]:evaluate()			-- set flag for dropout
        lst = clones.rnn[t]:forward(rnninp)	-- do one forward tick
        predictions[t] = lst
      end

      -- update hidden state
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

      -- prediction for t (t+1)
      stateLocs[t], stateLocs2[t], stateDA[t], DAsum[t], DAsum2[t], stateEx[t], smoothnessEx[t] = decode(predictions, t)
      labels = valLabTab[seq]:clone()
      exlabels = valExTab[seq]:clone()
      local input, output = getPredAndGTTables(stateLocs[t], stateLocs2[t], stateDA[t], DAsum[t], DAsum2[t], stateEx[t], t, smoothnessEx[t])

      --       print(outLocs, exTar)


      local tloss = 0
      if not sameWinSize then
        -- 	print(input)
        -- 	print(output)

        tloss = protoscrit:forward(input, output)

      else
        tloss = clones.criterion[t]:forward(input, output)
      end
      loss = loss + tloss
    end
    local predTracks, predTracks2, predDA, predEx = decode(predictions)
    if miniBatchSize>1 then predDA=predDA:sub(1,maxTargets) end
    --     local predEx = nil
    if logifyDA then predDA = torch.log(predDA) end

    -- plotting
    if seq==plotSeq then
      if nClasses==maxDets+2 then
        predEx = torch.ones(maxTargets,T,1):float()-torch.exp(predDA:narrow(3,nClasses,1))
      end
      predEx = predEx:sub(1,maxTargets)

      local predLab = getLabelsFromLL(predDA, false) -- do not resolve with HA
      predLab = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), predLab, 2)

      print('-- Validation -- ')
      eval_val_multass = printDA(predDA, predLab, predEx)
      plotProgress(tracks, detections, predTracks, predDA, predEx, 3, 'Validation', predTracks2)
      eval_val_mm = getDAError(predDA, labels:sub(1,maxTargets))

    end


  end
  loss = loss / T / tL	-- make average over all frames

  opt=deepcopy(tmpOpt)
  --     miniBatchSize=opt.mini_batch_size

  return loss
end


function eval_benchmark()
  return 1
end


-------------------------------------
--- LOSS AND GRADIENT COMPUTATION ---
-------------------------------------
-- loss function for temp window
local permPlot = math.random(tabLen(trTracksTab)) -- always plot this training seq
-- permPlot = 1
-- print(trTracksTab[permPlot])
-- print(trDetsTab[permPlot])
-- print(trOrigDetsTab[permPlot])
-- print(permPlot)

seqCnt=0
-- print(tabLen(trTracksTab))
-- abort()

function feval()
  grad_params:zero()
  tL = tabLen(trTracksTab)
  local randSeq = math.random(tL)	-- pick a random sequence from training set
  seqCnt=seqCnt+1
  if seqCnt > tabLen(trTracksTab) then seqCnt = 1 end
  randSeq = seqCnt

  permPlot = math.random(tabLen(trTracksTab)) -- randomize plot
  if (globiter % opt.plot_every) == 0 then randSeq = permPlot end -- force same sequence for plotting


  --   print(randSeq)
  --   print(trLabTab[randSeq])
  --   print(tabLen(trLabTab))

  tracks = trTracksTab[randSeq]:clone()
  detections = trDetsTab[randSeq]:clone()
  labels = trLabTab[randSeq]:clone()
  exlabels = trExTab[randSeq]:clone()

  if stateVel then vels = trVelTab[randSeq]:clone() end

  --   detections, labels = reshuffleDetsAndLabels2(detections, labels)
  TRAINING = true
  --   TRAINING = false
  ----- FORWARD ----
  local initStateGlobal = clone_list(init_state)
  local rnn_state = {[0] = initStateGlobal}
  --   local predictions = {[0] = {[opt.statePredIndex] = detections[{{},{t}}]}}
  local predictions = {}
  local loss = 0
  local stateLocs, stateLocs2, stateDA, stateEx, DAsum, DAsum2 = {}, {}, {}, {}, {}, {}
  smoothnessEx = {}
  local T = opt.temp_win - opt.batch_size
  local GTDA = {}
  for t=1,T do
    clones.rnn[t]:training()			-- set flag for dropout
    if opt.freq_reshuffle~= 0 then
      detections, labels = reshuffleDetsAndLabels2(detections, labels, t)
    end

    local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)		-- get combined RNN input table
    --       print(t)
    --       print(rnninp)
    --     print(rnninp[3]:squeeze())
    --     sleep(2)
    --     abort()
    local lst = clones.rnn[t]:forward(rnninp)	-- do one forward tick
    --     print(lst)
    --     print(lst[2])
    --     print(lst[3])
    --     print(lst[4])
    --     print(lst[5])
    --     abort()
    --     print(opt.statePredIndex)
    predictions[t] = lst
    --     print(lst[3]:squeeze())
    --     print(lst[4]:squeeze())
    --     sleep(2)

    -- update hidden state
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    --     print(rnn_state[t])
    --     abort()

    -- prediction for t (t+1)
    stateLocs[t], stateLocs2[t], stateDA[t], DAsum[t], DAsum2[t], stateEx[t], smoothnessEx[t] = decode(predictions, t)
    --       print(predictions)
    --       abort()
    local input, output = getPredAndGTTables(stateLocs[t], stateLocs2[t], stateDA[t], DAsum[t], DAsum2[t], stateEx[t], t, smoothnessEx[t])
    --       print(input[1])
    --       print(output[1])
    --       print(input[2])
    --       print(output[2])
    --       print(t)

    --       abort()
    --
    --       local diffPred = input[opt.exPredIndex]:squeeze()
    --       for k,v in pairs(stateEx) do
    -- 	print(k,stateEx[k]:squeeze(),smoothnessEx[k]:squeeze())
    --       end
    -- --       print(stateEx)
    -- --       print(smoothnessEx)
    -- --       print(smoothnessEx[t])
    --       if diffPred>0.5 then
    --       print(input[opt.exPredIndex])
    --       print(output[opt.exPredIndex])
    --       sleep(1)
    --       end
    --       abort()

    print(input[1])
    print(output[1])
    local tloss = clones.criterion[t]:forward(input, output)
    --     print(tloss)
    loss = loss + tloss
  end
  --   sleep(.1)
  loss = loss / T	-- make average over all frames
  local predTracks, predTracks2, predDA, predEx = decode(predictions)
  if miniBatchSize>1 then predDA=predDA:sub(1,maxTargets) end

  --   print(predDA)
  --   abort()

  -- plotting
  if (globiter == 1) or (globiter % opt.plot_every == 0) then
    if nClasses==maxDets+2 then
      predEx = torch.ones(maxTargets,T,1):float()-torch.exp(predDA:narrow(3,nClasses,1))
    end

    if logifyDA then predDA = torch.log(predDA) end



    predEx = predEx:sub(1,maxTargets)

    local predLab = getLabelsFromLL(predDA, false) -- do not resolve with HA
    predLab = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), predLab, 2)

    print('-- Training -- ')
    feval_multass = printDA(predDA, predLab, predEx)

    --     abort()
    plotProgress(tracks, detections, predTracks, predDA, predEx, 1, 'Training',predTracks2)
    feval_mm = getDAError(predDA, labels:sub(1,maxTargets))
    --     print(predDA)

    --     print(labels)
    --     print(feval_mm)
    --     abort()
  end

  ------ BACKWARD ------
  local rnn_backw = {}
  -- gradient at final frame is zero
  local drnn_state = {[T] = clone_list(init_state, true)} -- true = zero the clones
  for t=T,1,-1 do
    local input, output = getPredAndGTTables(stateLocs[t], stateLocs2[t], stateDA[t], DAsum[t], DAsum2[t], stateEx[t], t, smoothnessEx[t])

    --     print(input)
    --     print(output)

    local dl = clones.criterion[t]:backward(input,output)
    --     print(dl)
    local nGrad = #dl

    --     abort()
    --     for dd=1,nGrad do print(dd) print(dl[dd]) end


    --
    --     table.insert(drnn_state[t], dl) -- gradient of loss at time t
    for dd=1,nGrad do
      --       print('dd '..dd)
      --       local doInsert = true
      --       if predLoss and dd==opt.statePredIndex2 then doInsert = false end
      --       if doInsert then
      table.insert(drnn_state[t], dl[dd]) -- gradient of loss at time t
      --       end
    end
    --     print(drnn_state[t])
    --     abort()

    local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)		-- get combined RNN input table

    --     print(t)
    --     print(rnninp)
    --     print(drnn_state[t])
    --     abort()

    dlst = clones.rnn[t]:backward(rnninp, drnn_state[t])
    --     print(dlst)
    --     if t<T then abort() end
    drnn_state[t-1] = {}
    local maxk = opt.num_layers+1
    if opt.model == 'lstm' then maxk = 2*opt.num_layers+1 end
    --     if exVar then maxk = 2*opt.num_layers
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





-- start optimization here

train_losses = {}
val_losses = {}
real_losses = {}
train_mm, val_mm, real_mm = {}, {}, {}
train_ma, val_ma, real_ma = {}, {}, {}
mot15_mota = {}
mot15_mota_nopos = {}
local optim_state = {learningRate = opt.lrng_rate, alpha = opt.decay_rate}
local glTimer = torch.Timer()
for i = 1, opt.max_epochs do
  local epoch = i
  globiter = i

  if i>1 and (i-1)%(2*opt.synth_training)==0 and opt.random_epoch~=0 then
    LOAD = false
    opt.doSave=false
    trTracksTab, trDetsTab, trLabTab, trExTab, trSeqNames =
      prepareData('train', trSeqTable, trSeqTable, false)
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

    train_ma[i] = feval_multass+1
    val_ma[i] = eval_val_multass+1

    local plot_loss_x, plot_loss = getLossPlot(i, opt.eval_val_every, train_losses)
    local plot_val_loss_x, plot_val_loss = getValLossPlot(val_losses)

    local plot_train_mm_x, plot_train_mm = getValLossPlot(train_mm)
    local plot_val_mm_x, plot_val_mm = getValLossPlot(val_mm)
    local plot_real_mm_x, plot_real_mm = getValLossPlot(real_mm)

    local plot_train_ma_x, plot_train_ma = getValLossPlot(train_ma)
    local plot_val_ma_x, plot_val_ma = getValLossPlot(val_ma)


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
    pm(string.format('%10s%10.5f%10.5f%10.5f','Best',minTrainLoss,	minValidLoss,	minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,	minValidLossIt,	minRealLossIt))


    local minTrainLoss, minTrainLossIt = torch.min(plot_train_mm,1)
    local minValidLoss, minValidLossIt = torch.min(plot_val_mm,1)
    local minRealLoss, minRealLossIt = torch.min(plot_real_mm,1)
    minTrainLoss=minTrainLoss:squeeze()      minTrainLossIt=minTrainLossIt:squeeze()*opt.eval_val_every
    minValidLoss=minValidLoss:squeeze()      minValidLossIt=minValidLossIt:squeeze()*opt.eval_val_every
    minRealLoss=minRealLoss:squeeze()        minRealLossIt=minRealLossIt:squeeze()*opt.eval_val_every

    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','MissDA','Training','Valid','Real'))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Current',feval_mm+1,eval_val_mm+1,eval_benchmark_mm+1))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Best',minTrainLoss,	minValidLoss,	minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,	minValidLossIt,	minRealLossIt))
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

    table.insert(lossPlotTab, {"Trng MA",plot_train_ma_x, plot_train_ma, 'points pt 1'})
    table.insert(lossPlotTab, {"Vald MA",plot_val_ma_x, plot_val_ma, 'points pt 3'})


    -- 	local minInd = math.min(1,plot_loss:nElement())
    local maxY = math.max(torch.max(plot_loss), torch.max(plot_val_loss), torch.max(plot_real_loss),
      torch.max(plot_train_mm), torch.max(plot_val_mm), torch.max(plot_real_mm))*2
    local minY = math.min(torch.min(plot_loss), torch.min(plot_val_loss), torch.min(plot_real_loss),
      torch.min(plot_train_mm), torch.min(plot_val_mm), torch.min(plot_real_mm))/2
    rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",
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
    local meanmetsNOPOS = torch.zeros(1,14)
    if opt.eval_mot15 ~= 0 then
      local timer = torch.Timer()
      local trackerScript = 'rnnTrackerBFSEP'
      -- 	if stateVel then trackerScript = 'rnnTrackerV' end
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
      plot_mota_x, plot_mota = getValLossPlot(mot15_mota)
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
        csvWrite(string.format('%s/mota_%.1f.txt',outDir,maxMOTA),torch.Tensor(1,1):fill(maxMOTA))	-- write cost matrix
        csvWrite(string.format('%s/bm.txt',outDir,maxMOTA),torch.Tensor(1,1):fill(maxMOTA))	-- write cost matrix

        -- all-time best
        if writeBestMOTA(maxMOTA, modelName, modelSign,'bestmotaDA.txt') then -- update all time best
          savefile = string.format('bin/bestmotaDA-%.1f_x.t7',maxMOTA)
          saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
        end

      else
        pm(string.format('Max MOTA (%.2f) was at iteration %d',maxMOTA,plot_mota_x[maxMOTAIt]),2)
      end




    end -- if eval_mot15


    printModelOptions(opt, modelParams) -- print parameters
  end

  if i == 1 or i % (opt.print_every*10) == 0 then
    printModelOptions(opt, modelParams)
    printTrainingHeadline()
  end -- headline

end

-- print(profTable)
local mtime = 0
print('-------------   PROFILING   INFO   ----------------')
print(string.format('%20s%10s%7s','function name','time','calls'))
for k,v in pairs(profTable) do
  if v[1] > 0.01 then
    print(string.format('%20s%10.2f%7d',k,v[1],v[2]))
  end
  mtime = mtime + v[1]
end
print(string.format('%20s%10.2f%7s','total time meas.',mtime,''))
print(string.format('%20s%10.2f%7s','total time',ggtime:time().real,''))
