--     Online Multi-Target Tracking Using Recurrent Neural Networks
--     A. Milan, S. H. Rezatofighi, A. Dick, I. Reid, K. Schindler. In: AAAI 2017


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.misc'

require 'auxBFPAR'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a trajectory  model')
cmd:text()
cmd:text('Options')
-- main options
cmd:option('-model_name','rnnTracker','main model name')
cmd:option('-model_sign','r300_l1_n1_m1_d4','model signature')
cmd:option('-M',20,'Max number of total detections per frame')
cmd:option('-N',0,'Max number of max targets per frame. (0=based on max dets)')
cmd:option('-seq_name','art','Sequence Name')
cmd:option('-length',50,'number of frames to sample')
cmd:option('-gtdet',0,'use ground truth detections')
cmd:option('-normalize_data', 1, 'normalize data (zero-mean, std-dev 1')
cmd:option('-clean_res', 0, 'remove potential false positives')
cmd:option('-reset_state', 1, 'reset state each <tmp_win> frames')
cmd:option('-detfile', '', 'detections file')
cmd:option('-da', 4, '0=Hungarian, 1=GT, 2=learned')
cmd:option('-pred', 0, '0=learned, 1=GT')
cmd:option('-maxProb', 0, '0=raw probability, 1=max for each target')
cmd:option('-resHA',0 , 'resolve identities with HA')
cmd:option('-fixTracks',1 , 'show fixed tracks')
cmd:option('-showEx',0 , 'show existence')
-- further options
cmd:option('-profiler',0,'profiler on/off')
cmd:option('-verbose',2,'Verbosity level')
cmd:option('-suppress_x',0,'suppress plotting in terminal')
cmd:option('-eval',1,'evaluate if possible')
cmd:option('-crop_gt',0,'ajust ground truth to result length')
cmd:option('-plot_dim',1,'which dimention to plot')
cmd:option('-correct_dets',0,'trim detections with GT')
cmd:option('-get_det_trick', 0, 'get detections based on predictions')
cmd:option('-dummy_weight', 5, 'weight for assigning dummy det')
cmd:option('-model_suffix', '', 'suffix to model name (_val, _mota...)')
cmd:option('-use_pos', 1, 'use state, or detections')
cmd:option('-use_KF', 0, 'use kalman filter on DA')
cmd:option('-fframe', 1, 'start from this frame')
cmd:option('-ex_thr', 0.5, 'theshold for target existence')
cmd:option('-exConfThrFinal', 0, 'Final thr. no effect here')
cmd:option('-da_model','LSTM-DA','model name for Data Association')
-- artificial data options
cmd:option('-det_noise',0.01,'Detection noise std dev')
cmd:option('-det_fail',0.1,'Detection failure rate')
cmd:option('-det_false',0.2,'Detection false alarm rate')
cmd:option('-detConfThr',1e-5,'Prune threshold for det confidence')
cmd:option('-minLife',0,'Prune track lengths')
cmd:option('-seed',12,'Random seed')


cmd:text()
-- parse input params
sopt = cmd:parse(arg)

-- abort()
torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(sopt.seed)

-- local suffix = ''
-- suffix = '_val'
-- suffix = '_mota'
-- concat model name

if sopt.detfile=='' then sopt.detfile = nil end
sopt.model = getRNNTrackerRoot()..'bin/'..sopt.model_name
if sopt.model_sign ~= '' then sopt.model = sopt.model..'_'..sopt.model_sign end
sopt.model = sopt.model..sopt.model_suffix..'.t7'


-- load the model checkpoint
if not lfs.attributes(sopt.model, 'mode') then
  print('Error: File ' .. sopt.model .. ' does not exist.?')
end
if lfs.attributes('/home/h3/','mode') then sopt.suppress_x=1 end
print('Loading model ... '..sopt.model)
checkpoint = torch.load(sopt.model)


protos = checkpoint.protos
protos.rnn:evaluate()
opt = checkpoint.opt
opt.use_gt_input=0
opt.real_dets = 1-sopt.gtdet
opt.real_data = 1
mtype = opt.mtype
miniBatchSize = 1 or miniBatchSize
miniBatchSize = 1
opt.mini_batch_size = 1
opt.gpuid = -1
opt.synth_training = 1
opt.synth_valid = 1
opt.ex_thr = sopt.ex_thr

if sopt.det_noise~=0 then opt.det_noise = sopt.det_noise end
if sopt.det_fail~=0 then opt.det_fail = sopt.det_fail end
if sopt.det_false~=0 then opt.det_false = sopt.det_false end
opt.det_noise = sopt.det_noise
opt.det_fail = sopt.det_fail
opt.det_false = sopt.det_false

realData = true
if sopt.seq_name == 'art' then opt.real_data = 0; realData = false end
if opt.model==nil then opt.model='lstm' end

-- create some directories if they do not exist
createAuxDirs()


TRAINING = true
TRAINING = false
TESTING = true


--- ### --- ###--- ### --- ###--- ### --- ###--- ### --- ###--- ### --- ###
opt.trim_tracks  = 1 print('------ T R I M M I N G !!! ----------')
--- ### --- ###--- ### --- ###--- ### --- ###--- ### --- ###--- ### --- ###

getDetTrick = sopt.get_det_trick ~= 0
if getDetTrick then print('GET DET TRICK SET') end

if realData then
  local gttracks = getGTTracks(sopt.seq_name)
  if sopt.length == 0 then sopt.length = gttracks:size(2); opt.temp_win = sopt.length end
end


checkCuda()

local train_temp_win = opt.temp_win
pm('Training was performed on temp. windows of length '..train_temp_win, 3)
opt.temp_win = sopt.length
opt.suppress_x = sopt.suppress_x
opt.verbose = sopt.verbose
stateDim = opt.state_dim or 1
-- print("State dimension: "..stateDim)
modelParams = {'model_index','rnn_size', 'num_layers','max_n','max_m','state_dim'} -- TODO remove global
updLoss = opt.kappa ~= 0
predLoss = opt.lambda ~= 0
daLoss = opt.mu ~= 0
exVar = opt.nu ~= 0

fullStateDim = stateDim if stateVel then fullStateDim = stateDim * 2 end

opt.modelParams = modelParams
printModelOptions(opt, modelParams)

-- opt.use_da_input = 0 -- Hun
-- opt.use_da_input = 1 -- GT
-- opt.use_da_input = 2 -- Predict
opt.use_da_input = sopt.da
-- if lfs.attributes('/home/h3/','mode') then opt.use_da_input=0 end
---------------------
-- LOAD  DA  NETWORK
---------------------
DAopt = deepcopy(opt)
if opt.use_da_input==2 then

  local daRNNfile = getCheckptFilename('testingBFDAONLY', opt, modelParams)
  -- daRNNfile = string.gsub(daRNNfile,'li0','li1') print('WARNING! FORCE LINP') sleep(0.5)
  daRNNfile = string.gsub(daRNNfile,'mt3','mt1') print('WARNING! FORCE LSTM') sleep(0.1)
  daRNNfile = getRNNTrackerRoot()..'bin/0301DAa-1_mt1_r500_l1_n5_m5_d4_b1_v0_li0.t7'
  daRNNfile = getRNNTrackerRoot()..'bin/0301DAb-9_mt1_r900_l1_n5_m5_d4_b1_v0_li0.t7'
  daRNNfile = getRNNTrackerRoot()..'bin/0301DAb-2_mt1_r200_l1_n5_m5_d4_b1_v0_li0.t7'
  -- daRNNfile = 'bin/testingBFDAONLY_mt1_r100_l1_n5_m5_d4_b1_v0_li0_val.t7'
  -- daRNNfile = 'bin/testingBFDAONLY_mt1_r100_l1_n2_m2_d1_b1_v0_li0_val.t7'
  daRNNfile = getRNNTrackerRoot()..'bin/testingBFDAONLY_mt1_r30_l1_n3_m3_d2_b1_v0_li0_val.t7' -- mlmc
  -- daRNNfile = 'bin/testingBFDAONLY_mt1_r31_l1_n3_m3_d2_b1_v0_li0_val.t7' -- mse
  -- daRNNfile = 'bin/testingBFDAONLY_mt1_r29_l1_n3_m3_d2_b1_v0_li0_val.t7' -- cnll

  local daDim = math.min(opt.state_dim,2)
  daRNNfile = getRNNTrackerRoot()..'bin/testingBFDAONLY_mt1_r500_l1_n'..opt.max_n..'_m'..opt.max_m..'_d'..daDim..'_b1_v0_li0_val.t7'
  -- daRNNfile = 'bin/0307DAc-'..opt.max_n..'_mt1_r500_l1_n'..opt.max_n..'_m'..opt.max_m..'_d2_b1_v0_li0_val.t7'
  -- -- -- if sopt.da_model ~= nil then
  -- -- --   local i, t, popen = 0, {}, io.popen
  -- -- --   local pfile = popen('ls bin/'..sopt.da_model..'*_n'..opt.max_n..'_m'..opt.max_m..'_d2*_val*')
  -- -- --   print('looking for DA modules...')
  -- -- --   for filename in pfile:lines() do
  -- -- --     i = i + 1
  -- -- --     t[i] = filename
  -- -- --     print(filename)
  -- -- --   end
  -- -- --   if i>0 then
  -- -- --     daRNNfile = t[1]
  -- -- --   end
  -- -- -- end


  if onNetwork() then
    local parrun = opt.max_n
    daRNNfile = getRNNTrackerRoot()..'bin/0307DAc-'..parrun..'_mt1_r500_l1_n'..opt.max_n..'_m'..opt.max_m..'_d2_b1_v0_li0_val.t7'
    if opt.da_model ~= nil then
      daRNNfile = getRNNTrackerRoot()..'bin/'..opt.da_model..'c-'..parrun..'_mt1_r500_l1_n'..opt.max_n..'_m'..opt.max_m..'_d2_b1_v0_li0_val.t7'
    end
  end




  pm(string.format('Loading DA Module %s...',daRNNfile))



  if not lfs.attributes(daRNNfile, 'mode') then error('DA Module not found') end
  daCheckpoint = torch.load(daRNNfile)
  DAprotos = daCheckpoint.protos
  DAprotos.rnn:evaluate()
  DAopt = daCheckpoint.opt

  DAexVar = DAopt.nu ~= 0

  opt.pwd_mode = DAopt.pwd_mode

elseif opt.use_da_input == 3 then
  daRNNfile= getRNNTrackerRoot().. 'bin/testingHUN_mt1_r150_l1_n5_m5.t7'
  if sopt.da_model ~= nil then
    daRNNfile = getRNNTrackerRoot()..'bin/'..sopt.da_model..'.t7'
  end

  ------------------------------


  pm(string.format('Loading DA Module %s...',daRNNfile))
  if lfs.attributes(daRNNfile, 'mode') then
    checkpointHUN = torch.load(daRNNfile)
    protosHUN = checkpointHUN.protos
    protosHUN.rnn:evaluate()
    HUNopt = checkpointHUN.opt
    nClassesHUN=opt.max_m+opt.max_n
    pwdDimHUN = HUNopt.state_dim
    if HUNopt.pwd_mode == 0 then pwdDimHUN = 1 end -- euclidean
  else error('DA Module ' .. daRNNfile .. ' not found')
  end
  ------------------------------


else
  -- print(DAexVar)
  -- abort()

  DAopt = deepcopy(opt)
end



-- the initial state of the cell/hidden states
init_state = getInitState(opt, miniBatchSize)
DAinit_state = getDAInitState(opt, miniBatchSize)

-- print(DAinit_state)
-- abort()

local colorDetsShuffled = false
local colorDetsShuffled = true

------------------------------------------------
-------- Scene Info ----------------------------
local seqName = sopt.seq_name
maxRTargets, maxDets = opt.max_n, opt.max_m
maxFTargets = opt.max_nf
maxTargets = maxRTargets+maxFTargets
nClasses = maxDets + 1
xSize = stateDim*maxTargets
dSize = stateDim*maxDets
fullxSize = fullStateDim*maxTargets






-- tracks = getGTTracks(seqName)
-- detections = getDetTensor(seqName,sopt.detfile)

valSeqTable = {seqName}
realSeqNames = {}
table.insert(realSeqNames, valSeqTable) -- [1][1] is for [instance][minibatch]

-- print(valSeqTable)
imSizes = getImSizes(valSeqTable)


-- print(valSeqTable, maxTargets, maxDets)
local correctDets = sopt.correct_dets~=0
if correctDets then print('--- WARNING!!!! DETECTIONS ARE TRIMMED WITH GT ---') end

if seqName == 'art' then
  local trainingSequences = {'TUD-Campus','TUD-Stadtmitte'}
  opt.synth_training = 1
  opt.gtdet = 1
  imSizes = getImSizes(trainingSequences, imSizes)
  realTracksTab, realDetsTab, realLabTab, realExTab, realDetExTab, realSeqNames = prepareData('validation', valSeqTable, trainingSequences,  true)
else
  realTracksTab, realDetsTab, realLabTab, realExTab, realDetExTab, realSeqNames = prepareData('real', valSeqTable, valSeqTable,  true)
end
--   realTracksTab, realDetsTab, realLabTab =
--     getTracksAndDetsTables(valSeqTable, maxDets, maxDets, nil, correctDets,sopt.detfile)

-- printAll(realTracksTab[1], realDetsTab[1], realLabTab[1], realExTab[1], realDetExTab[1]);
-- abort()

unnormDetsTab={}
for k,v in pairs(realDetsTab) do unnormDetsTab[k] = realDetsTab[k]:clone() end


-- ALL DETS
-- if getDetTrick then
if seqName ~= 'art' then
  maxAllDets = sopt.M
  maxAllTargets = sopt.N

  local n,m = opt.max_n, opt.max_m
  maxTargets, maxDets = maxAllTargets, maxAllDets
  maxTargets = math.max(maxTargets,1)
  opt.max_n, opt.max_m = maxTargets, maxDets
  AllTracksTab, AllDetsTab, AllLabTab, AllExTab, AllDetExTab  = prepareData('real', valSeqTable, valSeqTable,  true)
  opt.max_n, opt.max_m = n,m
  maxTargets, maxDets = opt.max_n, opt.max_m

  AllunnormDetsTab={}
  for k,v in pairs(AllDetsTab) do AllunnormDetsTab[k] = AllDetsTab[k]:clone() end

  -- whats the max det per frame?
  if maxAllTargets<=0 then

    local maxAllDetsPerFrame =  torch.max(torch.sum(AllDetExTab[1],1))
    maxAllTargets = maxAllDetsPerFrame
    sopt.M = maxAllTargets
    print('Setting N = '..maxAllTargets)
  end
  nClassesHUN=sopt.M+sopt.N

  -- abort()


  --   getTracksAndDetsTables(valSeqTable, maxAllDets, maxAllDets, nil, false) -- last param: correct dets
  -- pm('Normalizing data...',2)
  -- AllDetsTab = normalizeData(AllDetsTab, AllDetsTab, false, maxAllDets, maxAllDets, realSeqNames, miniBatchSize==1)
  -- FLAG ALL DETECTIONS THAT REALLY EXIST
  -- AllDetExTab = {}
  --   for k,v in pairs(AllDetsTab) do
  --     AllDetExTab[k] = torch.zeros(maxAllDets*miniBatchSize, opt.temp_win):int()
  --     local detEx = v:narrow(3,1,1):reshape(maxAllDets*miniBatchSize,opt.temp_win):ne(0)
  --     AllDetExTab[k][detEx] = 1
  --   end
  --   print(AllDetExTab)
  --   abort()
  -- printAll(realTracksTab[1], realDetsTab[1], realLabTab[1], realExTab[1], realDetExTab[1]);
  -- printAll(AllTracksTab[1], AllDetsTab[1], realLabTab[1], realExTab[1], AllDetExTab[1]);
  -- abort()
else
  AllDetsTab, AllDetExTab = {}, {}
  for k,v in pairs(realDetsTab) do AllDetsTab[k] = v:clone() end
  for k,v in pairs(realDetExTab) do AllDetExTab[k] = v:clone() end
  maxAllDets = maxDets
  maxAllTargets = maxTargets
end

-- end

-- print(AllDetsTab[1])
-- abort()


function getLocFromFullState(state)
  -- remove velocities estimations
  if opt.vel~=0 then
    local ind = torch.linspace(1,fullStateDim-1,fullStateDim/2):long()
    state = state:index(3,ind)
  end
  return state
end


function plotProgress(tracks, detections, predTracks, predDA, predEx, winID, winTitle)
  predDA = predDA:sub(1,maxTargets):float()
  if exVar then predEx = predEx:sub(1,maxTargets):float() else predEx = nil end

  local DA = getLabelsFromLL(predDA)
  plotTab={}

  local N,F,D=getDataSize(detections:sub(1,maxDets))
  local da = torch.IntTensor(N,F)
  if colorDetsShuffled then
    for i=1,maxDets do da[i]=(torch.ones(1,opt.temp_win)*i) end -- color dets shuffled
  end
  plotTab = getTrackPlotTab(tracks:sub(1,maxTargets):float(), plotTab, 1)
  plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, da)
  plotTab = getTrackPlotTab(predTracks:sub(1,maxTargets):float(), plotTab, 2, nil, predEx, 1, DA)
  plot(plotTab, winID, string.format('%s',winTitle), nil, 0) -- do not save first
  sleep(.1)
end


------------   MAIN  PREDICTION   -----------
-- ####################################### --
local initStateGlobal = clone_list(init_state)
for k,v in pairs(initStateGlobal) do initStateGlobal[k] = v:sub(1,1) end


local T = opt.temp_win - opt.batch_size

local rnn_state = {[0] = initStateGlobal}
local Allrnn_states = {}
for tar=1,maxAllTargets do
  Allrnn_states[tar] = {}
  for t=1,T do Allrnn_states[tar][t] = {} end
end
for tar=1,maxAllTargets do
  for t=1,T do Allrnn_states[tar][t] = {} end
  Allrnn_states[tar][0] = initStateGlobal
end


local predictions = {}
local Allpredictions = {}
for tar=1,maxAllTargets do
  Allpredictions[tar] = {}
  for t=1,T do
    Allpredictions[tar][t] = {}
    for k=1, 6 do
      Allpredictions[tar][t][k] = {}
    end
  end
end
-- print(Allpredictions)
-- Allpredictions[1][1][2]=2
-- print(Allpredictions)
-- print(Allpredictions)
-- abort()

local predictionsTemp = {}
local stateLocs, stateLocs2, stateDA, stateEx, statePred = {}, {}, {}, {}, {}

local AllstateLocs, AllstateLocs2, AllstateDA, AllstateEx, AllstatePred = {}, {}, {}, {}, {}
for tar=1,maxAllTargets do
  AllstateLocs[tar]={} for t=1,T do AllstateLocs[tar][t] = {} end
  AllstateLocs2[tar]={}for t=1,T do AllstateLocs2[tar][t] = {} end
  AllstateDA[tar]={}for t=1,T do AllstateDA[tar][t] = {} end
  AllstateEx[tar]={}for t=1,T do AllstateEx[tar][t] = {} end
  AllstatePred[tar]={}for t=1,T do AllstatePred[tar][t] = {} end
end

local T = opt.temp_win - opt.batch_size
tracks = realTracksTab[1]
detections = realDetsTab[1]:clone()
labels = realLabTab[1]:clone() -- only for debugging
exlabels = realExTab[1]:clone()
detexlabels = realDetExTab[1]:clone()
alldetections = AllDetsTab[1]:clone()
alldetexlabels = AllDetExTab[1]:clone()

if getDetTrick then alldetections = AllDetsTab[1]:clone() end
-- print(detections)
-- abort()
-- opt.use_gt_input=1 print('USING GT!!!') sleep(.5)

-----------------------------

-- initialize hidden state with zeros
local DAinitStateGlobal = clone_list(DAinit_state)
local DArnn_state = {[0] = DAinitStateGlobal}
local DApredictions = {}
local allPredDA = torch.zeros(maxAllTargets,T,maxAllDets+1)
local allPredEx = torch.zeros(maxTargets, T, 1)
local pclone=protos.rnn:clone()

local protosClones = {}
for tar=1,maxAllTargets do
  table.insert(protosClones, protos.rnn:clone())
end
-- print(protosClones[1])
-- print(protos.rnn)
-- -- abort()

DAtmpStateEx = torch.ones(maxTargets):float()
DAtmpDA = torch.zeros(maxTargets, nClasses):float()
DAtmpDAsum = torch.zeros(maxTargets, 1):float()
for t=1,T do
  -- if we want to reset hidden state periodically
  if sopt.reset_state>0 and ((t-1) % torch.round(train_temp_win/1*1) == 0) then
    --     rnn_state = {[t-1] = initStateGlobal}
    for tar=1,maxAllTargets do
      --       table.insert(Allrnn_states, rnn_state)
      Allrnn_states[tar][t-1] = initStateGlobal
    end
  end

  if ((t-1) % torch.round(train_temp_win/1*1) == 0) then
    DArnn_state = {[t-1] = DAinitStateGlobal}
  end

  --- PREDICT  STATE ---
  -- dummy DA and existence
  --   detIDforUpdate=torch.ones(maxTargets,1)

  --
  if opt.use_da_input == 2 then
    local rnninp, _ = getRNNInput(t, rnn_state, predictions)		-- get combined RNN input table
    local Predlst = pclone:forward(rnninp)	-- do one forward tick
    predictionsTemp[t] = {}
    for k,v in pairs(Predlst) do predictionsTemp[t][k] = v:clone() end -- deep copy
    _, statePred = decode(predictionsTemp, t)  -- get predicted state
    if sopt.pred == 1 then
      statePred = tracks[{{},{t+1}}]:clone()
      print('WARNING! Use GT as prediction')
    end


    --     print(rnninp[1]:reshape(maxTargets,stateDim))
    --     print(t)
    --     print(statePred:reshape(maxTargets,stateDim))
    --     sleep(2)

    if getDetTrick then
      local pwdDim = math.min(2,stateDim)

      local maxDist = 0.1
      local inpPred = statePred:reshape(maxTargets,stateDim):narrow(2,1,pwdDim)
      --       print(inpPred)
      local det_allx = alldetections[{{},{t+1},{1,pwdDim}}]:reshape(maxAllDets,pwdDim);
      local det_x = detections[{{},{t+1},{1,pwdDim}}]:reshape(maxDets,pwdDim);
      --       print(det_allx)


      local allDist = torch.ones(maxTargets,maxAllDets) * 100
      for tar=1,maxTargets do
        for det=1,maxAllDets do
          if det_allx[det][1] ~= 0 then
            allDist[tar][det] = torch.dist(inpPred[tar], det_allx[det])
          end
        end
      end
      print(allDist)
      local mv, mi = torch.min(allDist,2) mv = mv:reshape(maxTargets) mi = mi:reshape(maxTargets)
      print(mv)

      --       print(detections)
      for tar=1,maxTargets do
        if mv[tar] < maxDist and alldetections[mi[tar]][t+1]~=0 then
          -- 	  print('take '..mi[tar])
          detections[tar][t+1] = alldetections[mi[tar]][t+1]:clone()
          alldetections[mi[tar]][t+1] = 0
        end
      end
      --       print(detections)
    end

    -- Do Data Association --
    local DArnninp, DArnn_state = getDARNNInput(t,DArnn_state, DApredictions, statePred)
    --     print(DArnninp)
    local DAlst = DAprotos.rnn:forward(DArnninp)
    DApredictions[t] = {}
    for k,v in pairs(DAlst) do DApredictions[t][k] = v:clone() end -- deep copy

    DAtmpState, DAtmpState2, DAtmpDA, DAtmpDAsum, DAtmpStateEx  = DAdecode(DApredictions, t)
    --         print(DAtmpDA)
    DAtmpDA = torch.log(DAtmpDA) -- HACK LOGIFY
    DAtmpDA = DAtmpDA:reshape(maxTargets, nClasses)

    --     print(torch.exp(DAtmpDA))
    --     abort()
    --normalize for plotting

    --     local takeMaxProb = true
    --     local takeMaxProb = false
    local takeMaxProb = sopt.maxProb ~= 0

    --     DAtmpDA = torch.exp(DAtmpDA)
    --     DAtmpDA = torch.pow(DAtmpDA,6)
    --     DAtmpDA = torch.log(DAtmpDA)
    if DAopt.bce~=0 then
      --       DAtmpDA = torch.log(DAtmpDA)
      --       DAtmpDA = torch.pow(DAtmpDA,12)
      for tar=1,maxTargets do
        local probLine = torch.exp(DAtmpDA[tar]:reshape(nClasses))
        probLine = probLine / torch.sum(probLine)
        -- 	print(torch.exp(probLine))

        if takeMaxProb then
          local mv, mi = torch.max(torch.exp(probLine):reshape(1,nClasses),2)
          -- 	  print(mv,mi)
          mv=mv:squeeze()
          mi=mi:squeeze()
          -- 	  print(mv)
          -- 	  print(mi)

          probLine:fill(0)
          probLine[mi] = 1
          probLine = makePseudoProb(probLine:reshape(nClasses),.0001)
        end


        probLine = torch.log(probLine)
        DAtmpDA[tar] = probLine
      end

    else
      --       DAtmpDA = torch.log(torch.pow(torch.exp(DAtmpDA),12))

      --       print(torch.exp(DAtmpDA))
      local doNormalize = true
      --       local doNormalize = false
      if doNormalize then
        -- normalize
        for tar=1,maxTargets do
          local probLine = torch.exp(DAtmpDA[tar]:reshape(nClasses))
          probLine = probLine / torch.sum(probLine)

          probLine = torch.log(probLine)
          DAtmpDA[tar] = probLine
        end
      end
      --       print(torch.exp(DAtmpDA))
      --       abort()

      if takeMaxProb then
        for tar=1,maxTargets do
          local probLine = torch.exp(DAtmpDA[tar]:reshape(nClasses))
          probLine = probLine / torch.sum(probLine)
          local mv, mi = torch.max(torch.exp(probLine):reshape(1,nClasses),2)
          mv=mv:squeeze()  mi=mi:squeeze()
          probLine:fill(0)
          probLine[mi] = 1
          probLine = makePseudoProb(probLine:reshape(nClasses),.001)
          probLine = torch.log(probLine)
          DAtmpDA[tar] = probLine

        end


      end

    end


    ---- resolve with Hungarian
    local doHA = sopt.resHA ~= 0
    --     print(doHA)
    --     abort()
    --     local doHA = false

    if doHA then
      --       abort()
      local missThr = 0.05;
      local allDist=DAtmpDA * (-1)
      if maxTargets>1 then
        allDist=allDist:cat(torch.FloatTensor(maxTargets,maxTargets-1):fill(torch.exp(missThr)))
      end

      --     print(allDist)
      local status, ass = pcall(hungarianL,allDist)
      if status then


        local lab = torch.zeros(maxTargets,nClasses):float()
        lab:fill(0)
        for tar=1,maxTargets do
          local assClass = ass[tar][2]
          if assClass>nClasses then assClass=nClasses end
          lab[tar][assClass] = 1
          lab[tar] = makePseudoProb(lab[tar]:reshape(nClasses):float(),.001)
        end
        DAtmpDA = torch.log(lab)
      else
        print('WARNING! HA failed!')
      end
    end

    --     print(torch.exp(globalLAB))
    --     print(torch.exp(globalLAB):lt(0.5))
    --     print(torch.exp(DAtmpDA))
    --     print(torch.exp(DAtmpDA):lt(.5))
    --     sleep(2)
    globalLAB = DAtmpDA:clone()

    -- update hidden state DA
    DArnn_state[t] = {}
    for i=1,#DAinit_state do table.insert(DArnn_state[t], DAlst[i]) end -- extract the state, without output

    --       abort()
  end



  Allrnninps, Allrnn_states = getRNNInput(t, Allrnn_states, Allpredictions)		-- get combined RNN input table


--  print(allPredDA)
--  print(globalLAB[i])
  for i=1,maxAllTargets do allPredDA[{{i},{t},{}}] = globalLAB[i]:clone() end
  -- 	for i=1,maxTargets do allPredEx[{{i},{t},{}}] = globalEXLAB[i] end

  --       print(torch.exp(globalLAB))
  --       abort()

  --       for i=1,maxTargets do allPredDA[{{i},{t},{}}] = globalLAB[i]:clone() end


  local allLst={}
  for tar=1,maxAllTargets do
    -- 	print(tar)
    local rnninp = {}
    -- 	print(Allrnninps[tar])
    for i=1,#Allrnninps[tar] do table.insert(rnninp, Allrnninps[tar][i]:clone()) end
    -- 	print(rnninp[1])
    -- 	print(rnninp[3])
    -- 	abort()

    local lst =protosClones[tar]:forward(rnninp)	-- do one forward tick

    -- 	print('for target '..tar)
    -- 	print('prediction ') print(lst[3])
    -- 	print('update ') print(lst[2])
    table.insert(allLst, lst)
  end


  local tmpRNNstates = {}
  for tar=1,maxAllTargets do
    for k,v in pairs(allLst[tar]) do Allpredictions[tar][t][k] = v:clone() end -- deep copy

    local rnn_state = {}
    for i=1,#init_state do
      table.insert(rnn_state, allLst[tar][i])
    end -- extract the state, without output
    table.insert(tmpRNNstates, rnn_state)
  end


  --       print(tmpRNNstates[1][1])
  --       print(tmpRNNstates[2][1])

  --       for tt=1,t do
  -- 	print(Allpredictions[1][tt][2])
  --       end
  --       sleep(1)
  for tar=1,maxAllTargets do
    Allrnn_states[tar][t] = deepcopy(tmpRNNstates[tar])
  end
  --       print(Allrnn_states[1][1][1]:sub(1,1,1,5))
  --       print(Allrnn_states[2][1][1]:sub(1,1,1,5))

  for tar=1,maxAllTargets do
  -- 	print(Allpredictions[tar][t][2])
  -- 	sleep(1)
  end

  -- abort()

  predictions = moveState(Allpredictions, t)
  -- update hidden state


  -- prediction for t (t+1)
  --       stateLocs[t], stateLocs2[t], stateDA[t] = decode(predictions, t)
  local A,B,C = decode(Allpredictions, t)

  for tar=1,maxAllTargets do
    AllstateLocs[tar] = A[tar]:clone()
    AllstateLocs2[tar] = B[tar]:clone()
  end
  --       print('update')
  --       print(A[1])
  --       abort()
  --       print(A[2])
  --       abort()



end
-- ####################################### --
-- abort()
-- local predTracks, predTracks2, predDA, predEx = decode(predictions)
local AllpredTracks, AllpredTracks2, allpredDA, allpredEx = decode(Allpredictions)
-- abort()
-- print(AllpredTracks[1])
-- abort()
-- print(AllpredTracks[2])
-- abort()

local predTracks = torch.zeros(1,T,stateDim)
local predTracks2 = torch.zeros(1,T,stateDim)
local predDA = torch.zeros(1,T,nClasses)
local predEx = torch.zeros(1,T,1)
for tar=1,maxAllTargets do
  predTracks = predTracks:cat(AllpredTracks[tar], 1)
  predTracks2 = predTracks2:cat(AllpredTracks2[tar], 1)
  predDA = predDA:cat(allpredDA[tar], 1)
  predEx = predEx:cat(allpredEx[tar], 1)
end
predTracks = predTracks:sub(2,-1)
predTracks2 = predTracks2:sub(2,-1)
predDA = predDA:sub(2,-1)
predEx = predEx:sub(2,-1)
-- print(predTracks)
-- abort()


-- predTracks = tracks:narrow(2,2,sopt.length-1):clone() print('USING GT!!!')
predDA=allPredDA:clone()
-- predEx = allPredEx:clone()

--- ### GT
-- predDA = getOneHotLabAll(labels, opt.mini_batch_size == 1):sub(1,1,2,opt.temp_win):float()
-- predDA = oneHotPseudoProb(predDA)
-- predDA = torch.log(predDA)
-- print(allPredDA)
-- abort()
-- _,_,predDA = DAdecode(DApredictions)

-- if not onNetwork() then
--   print('predicted tracks X')
--   printDim(predTracks,1)
--   if opt.state_dim>1 then
--   print('predicted tracks Y')
--   printDim(predTracks,2)
--   end
-- end

-- print(predDA)
-- print(predEx)
-- abort()


predTracks = getLocFromFullState(predTracks)
local predExBin=getPredEx(predEx)
--print(predExBin)
-- printDim(predTracks)
-- abort()
-- print(predTracks)


-- print(predLab)
-- print(sopt.resHA~=0)
predLab = getLabelsFromLL(predDA, false) -- true for Hungarian 1-to-1 assignment
-- print(predLab)
-- abort()
-- predLab = torch.cat(torch.linspace(1,maxAllTargets,maxAllTargets):int():reshape(maxAllTargets,1), predLab, 2)
-- local predLabRaw = getLabelsFromLL(predDA, false)
-- pad 1,2,3...

-- predLabRaw = torch.cat(torch.linspace(1,maxTargets,maxTargets):int():reshape(maxTargets,1), predLabRaw, 2)

-- print('Predicted DA')
-- print(predLabRaw)
-- print('Ground Truth DA')
-- print(labels)
-- print('Mispredicted')
-- local misDA = torch.abs(predLabRaw:float()-labels)
-- misDA[misDA:ne(0)]=1
-- print(misDA)
-- local N,F=getDataSize(labels)
-- print(string.format('Mispredicted labels: %d = %.2f %%',torch.sum(misDA), torch.sum(misDA)*100/(F*N)))

local finalTracks = predTracks:clone()
local finalTracks2 = predTracks2:clone()


-- printDim(finalTracks2)
-- print(predExBin)
-- abort()

-- print(finalTracks)
-- abort()

local N,F,D = getDataSize(finalTracks)
-- print(N,F,D)
-- abort()
local fixedTracks = torch.zeros(1,F,D)
local fixedTracks2 = torch.zeros(1,F,D)
local fixedEx = torch.zeros(1,F):int()
for tar=1,N do
  local started, finished=0,0
  for t=1,F do
    if (t==1 and predExBin[tar][t]==1) or (t>1 and predExBin[tar][t]==1 and predExBin[tar][t-1]==0) then
      started=t
    end
    if (t==F and predExBin[tar][t]==1) or (t<F and predExBin[tar][t]==1 and predExBin[tar][t+1]==0) then
      finished=t
    end
    if started>0 and finished>0 then
      local tmpTrack = torch.zeros(1,F,D)
      tmpTrack[{{1},{started,finished},{}}] = finalTracks[{{tar},{started,finished},{}}]
      fixedTracks=fixedTracks:cat(tmpTrack,1)

      tmpTrack[{{1},{started,finished},{}}] = finalTracks2[{{tar},{started,finished},{}}]
      fixedTracks2=fixedTracks2:cat(tmpTrack,1)

      started=0
      finished=0
      --       abort()
    end
  end
end

-- print(fixedTracks)
-- abort()
if fixedTracks:size(1)>1 then
  fixedTracks=fixedTracks:sub(2,-1)
  fixedTracks2=fixedTracks2:sub(2,-1)
else
  fixedTracks=finalTracks:clone()
  fixedTracks2=finalTracks2:clone()
end
--   fixedTracks=finalTracks:clone()
--   fixedTracks2=finalTracks2:clone()


-- finalTracks = fixedTracks:clone()
-- printDim(finalTracks)
-- abort()

-- if sopt.use_pos == 0 then finalTracks = getSnappedPos(finalTracks, detections) end
-- if sopt.use_pos == 0 then
-- end

-- predEx=nil
-- printDim(finalTracks)

local N,F,D=getDataSize(detections)
local da = torch.IntTensor(N,F)
if colorDetsShuffled then for i=1,maxDets do da[i]=i end end

-- zero out inexisting tracks
local ztracks = tracks:clone()
local N,F,D=getDataSize(tracks)
-- for i=1,N do for t=1,F do if exlabels[i][t]==0 then ztracks[i][t]=0 end end end
plotTab = {}
-- print(ztracks)
-- plotTab = getTrackPlotTab(ztracks, plotTab, 1)  -- ground truth

local trueDets = alldetexlabels:reshape(maxAllDets, opt.temp_win, 1):expand(maxAllDets, opt.temp_win, stateDim)
local virtDets = alldetexlabels:eq(0):reshape(maxAllDets, opt.temp_win, 1):expand(maxAllDets, opt.temp_win, stateDim)
local realDets = alldetections:clone():cmul(trueDets:float())
local virtDets = alldetections:clone():cmul(virtDets:float())

plotTab = getDetectionsPlotTab(realDets, plotTab, nil, nil)
if getDetTrick then plotTab = getDetectionsPlotTab(AllDetsTab[1], plotTab, nil) end
-- plotTab = getDetectionsPlotTab(virtDets, plotTab, nil, nil, 0, true) -- virtual dets

GTSTATE = tracks:clone()
if sopt.gtdet==1 or true then
  if sopt.seq_name == 'art' then
    --     print('WARNING! GT LABELS')
    --     predLab = labels:clone()

    --     predLab:zero()
    --     print(predDA)
    --     predDA = torch.ones(maxTargets, T, maxDets+1) * (-10)


    for t=1,T do
    --       detIDforUpdate = labels:narrow(2,t+1,1):squeeze():clone() -- GT labels
    --       print(detIDforUpdate)
    --       for i=1,maxTargets do predDA[i][t][detIDforUpdate[i]] = 0 end
    end
  end
  --   printDA(predDA, predLab, predEx)

  --   print(predDA)
  --   abort()
  --   plotTab = getDAPlotTab(finalTracks, detections, plotTab, predLab, predEx, 0, predDA)
  --   plotTab = getDAPlotTab(predTracks:sub(1,maxTargets):float(), detections:sub(1,maxDets):float(), plotTab, fullDA, predEx)
end


-- printDim(fixedTracks)
-- fixedTracks
-------------------- # - # - # ###---### --- ###

if sopt.fixTracks ~= 0 then
  --   plotTab = getTrackPlotTab(fixedTracks2, plotTab, 3,nil, nil, 1)  -- prediction fixed
  plotTab = getTrackPlotTab(fixedTracks, plotTab, 2,nil,nil,1)  -- update fixed
else
  --   plotTab = getTrackPlotTab(finalTracks2, plotTab, 3,nil, predEx, 1)  -- prediction
  plotTab = getTrackPlotTab(finalTracks, plotTab, 2,nil, predEx, 1) -- update
end

if sopt.showEx ~=0 then
  plotTab = getExPlotTab(plotTab, predEx, 1)
end
-- -- --------------------
plot(plotTab,1,'Final Result - '..seqName)


-- plot second dimension


--- Set inex. to 0
local N,F,D = getDataSize(finalTracks)
-- print(N,F,D)
-- abort()
for t=1,F do
  for i=1,N do
    if predExBin[i][t] == 0 then finalTracks[i][t] = 0  end
  end
end

if onNetwork() then
  for t=1,F do  for i=1,N do
    if predExBin[i][t] == 0 then finalTracks[i][t] = 0  end
  end end
end

-- denormalize and write out
-- realTracksTab = normalizeData(AllTracksTab, AllunnormDetsTab, false, maxAllTargets, maxAllDets, realSeqNames)

local origFinalTracks = finalTracks:clone() -- normalized
finalTracksTab={}
table.insert(finalTracksTab, finalTracks)

if realData then
  finalTracksTab = normalizeData(finalTracksTab, AllunnormDetsTab, true, maxAllTargets, maxAllDets, realSeqNames)
end
local writeResTensor = finalTracksTab[1]


-- move result forward in time if necessary
if sopt.fframe ~= nil and sopt.fframe>1 then
  local N,F,D = getDataSize(writeResTensor)
  local prePad = torch.zeros(N,sopt.fframe-1,D)
  writeResTensor=prePad:cat(writeResTensor, 2)
end


-- smash existence probability as dim = 5
writeResTensor = writeResTensor:cat(predEx, 3)

-- remove false tracks
writeResTensor = writeResTensor:sub(1,maxAllTargets)


local outDir = getResDir(sopt.model_name, sopt.model_sign)
mkdirP(outDir)
print(outDir)
writeTXT(writeResTensor, string.format("%s/%s.txt",outDir, seqName))
