require 'torch'
require 'nn'
require 'nngraph'


--------------------------------------------------------------------------
--- each layer has the same hidden input size
function getInitStateHUN(HUNopt, miniBatchSize)
  local init_state = {}
  for L=1,HUNopt.num_layers do
    local h_init = torch.zeros(miniBatchSize, HUNopt.rnn_size)
    table.insert(init_state, dataToGPU(h_init:clone()))
    if HUNopt.model == 'lstm' then
      table.insert(init_state, dataToGPU(h_init:clone()))
    end
  end
  return init_state
end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t		time step
-- @param rnn_state	hidden state of RNN (use t-1)
-- @param predictions	current predictions to use for feed back
function getRNNInputHUN(t, rnn_state, predictions)
  local rnninp = {}

  -- Cost matrix
  local loccost = costs:clone():reshape(miniBatchSize, maxTargets*nClassesHUN*pwdDimHUN)
  table.insert(rnninp, loccost)

  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end

  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions	current predictions to use for feed back
-- @param t		time step (nil to predict for entire sequence)
function decodeHUN(predictions, tar)
  local DA = {}
  local T = tabLen(predictions)	-- how many targets
  if tar ~= nil then
    local lst = predictions[tar]
    DA = lst[HUNopt.daPredIndex]:reshape(miniBatchSize, nClassesHUN) -- miniBatchSize*maxTargets x maxDets
  else
    DA = zeroTensor3(miniBatchSize,T,nClassesHUN)
    for tt=1,T do
      local lst = predictions[tt]
      DA[{{},{tt},{}}] = lst[HUNopt.daPredIndex]:reshape(miniBatchSize, 1, nClassesHUN)
    end
  end
  return DA
end



--------------------------------------------------------------------------
--- each layer has the same hidden input size
function getInitState(opt, miniBatchSize)
  local init_state = {}
  for L=1,opt.num_layers do
    local h_init = torch.zeros(miniBatchSize, opt.rnn_size)
    table.insert(init_state, dataToGPU(h_init:clone()))
    if opt.model == 'lstm' then
      table.insert(init_state, dataToGPU(h_init:clone()))
    end
  end
  return init_state
end

function moveState(predictions, t)
  if TRAINING then return predictions end

  ------- MOVE PREDICTED STATE AROUND ----------
  local exlab = {}
  --     if t==1 then exlab=torch.ones(maxTargets):float() * 0.5
  local movedState = nil
  if t>=1 and not TRAINING then
    local exThreshold = opt.ex_thr
    local stateLocs, stateLocs2, stateDA, exlab=decode(predictions,t)

    movedState = stateLocs:clone():reshape(maxTargets, stateDim)
    local det_x =	detections[{{},{t+1}}]:clone():reshape(maxDets,stateDim)




    exlab=exlab:reshape(maxTargets)
    stateDA = torch.exp(globalLAB)

    local unclaimedDets = torch.ByteTensor(maxDets):fill(0)
    local unclaimedTars = torch.ByteTensor(maxTargets):fill(0)
    --       print(unclaimedDets)
    for det=1,maxDets do
      if torch.max(stateDA:narrow(2,det,1)) < 0.5 and detexlabels[det][t+1]==1 then
        unclaimedDets[det]=1
      end
    end
    for tar=1,maxTargets do
      if exlab[tar]<exThreshold then
        unclaimedTars[tar]=1
      end
    end

    --       print('t '..t)
    --       print('lab ') print(stateDA)
    --       print('lab ') print(stateDA:lt(0.5))
    --       print('exlab ') print(exlab)
    --       print('uncTars ') print(unclaimedTars)
    --       print('uncDets ') print(unclaimedDets)
    --       sleep(1)
    local distNorm = 2
    local nDets, nTars = unclaimedDets:eq(1):sum(),unclaimedTars:eq(1):sum()

    -- GREEDY BASED ON DISTANCE
    local pwdDim = math.min(stateDim, 2)
    if nDets>0 and nTars>0 then
      for tar=1,maxTargets do
        if unclaimedTars[tar] == 1 then

          -- get all distances from target to dets
          local allDist = torch.ones(maxDets) * 10
          for det=1,maxDets do
            if unclaimedDets[det]==1 then
              allDist[det] = torch.dist(movedState[tar]:narrow(1,1,pwdDim), det_x[det]:narrow(1,1,pwdDim), distNorm)
            end
          end -- for det

          -- 	    print(tar)
          -- 	    print(allDist)
          -- which one is closest?
          local mv, det = torch.min(allDist,1)
          mv=mv:squeeze()
          det=det:squeeze()
          -- 	    print(mv,det)

          -- move to closest
          if mv<10 then
            unclaimedTars[tar]=0
            unclaimedDets[det]=0
            -- 	      print('Frame '..(t+1)..'. Moving target update '..tar..' to det '..det)
            movedState[tar] = det_x[det]:clone()
          end

        end -- if unclaimedTars
      end -- for tar
    end -- if ndets
    --       print(movedState)
    --       abort()


    --     print(t)
    --     print(inputState)
    --     abort()
    movedState = movedState:reshape(opt.mini_batch_size,xSize)
    predictions[t][opt.updIndex] = movedState:clone()
  end
  ------------------------------------

  return predictions
end

-- function moveStateAllDets(predictions, t)
--   if TRAINING then return predictions end
--
--     ------- MOVE PREDICTED STATE AROUND ----------
--     local exlab = {}
-- --     if t==1 then exlab=torch.ones(maxTargets):float() * 0.5
--     movedState = nil
--     if t>=1 and not TRAINING then
--       local exThreshold = opt.ex_thr
--       local stateLocs, stateLocs2, stateDA, exlab=decode(predictions,t)
--
--       movedState = stateLocs:clone():reshape(maxTargets, stateDim)
--       local det_x =	detections[{{},{t+1}}]:clone():reshape(maxDets,stateDim)
--
--
--
--
--       exlab=exlab:reshape(maxTargets)
--       stateDA = torch.exp(globalLAB)
--
--       local unclaimedDets = torch.ByteTensor(maxDets):fill(0)
--       local unclaimedTars = torch.ByteTensor(maxTargets):fill(0)
-- --       print(unclaimedDets)
--       for det=1,maxDets do
-- 	if torch.max(stateDA:narrow(2,det,1)) < 0.5 and detexlabels[det][t+1]==1 then
-- 	  unclaimedDets[det]=1
-- 	end
--       end
--       for tar=1,maxTargets do
-- 	if exlab[tar]<exThreshold then
-- 	  unclaimedTars[tar]=1
-- 	end
--       end
--
-- --       print('t '..t)
-- --       print('lab ') print(stateDA)
-- --       print('lab ') print(stateDA:lt(0.5))
-- --       print('exlab ') print(exlab)
-- --       print('uncTars ') print(unclaimedTars)
-- --       print('uncDets ') print(unclaimedDets)
-- --       sleep(1)
--       local distNorm = 2
--       local nDets, nTars = unclaimedDets:eq(1):sum(),unclaimedTars:eq(1):sum()
--
--       -- GREEDY BASED ON DISTANCE
--       local pwdDim = math.min(stateDim, 2)
--       if nDets>0 and nTars>0 then
-- 	for tar=1,maxTargets do
-- 	  if unclaimedTars[tar] == 1 then
--
-- 	    -- get all distances from target to dets
-- 	    local allDist = torch.ones(maxDets) * 10
-- 	    for det=1,maxDets do
-- 	      if unclaimedDets[det]==1 then
-- 		allDist[det] = torch.dist(movedState[tar]:narrow(1,1,pwdDim), det_x[det]:narrow(1,1,pwdDim), distNorm)
-- 	      end
-- 	    end -- for det
--
-- -- 	    print(tar)
-- -- 	    print(allDist)
-- 	    -- which one is closest?
-- 	    mv, det = torch.min(allDist,1)
-- 	    mv=mv:squeeze()
-- 	    det=det:squeeze()
-- -- 	    print(mv,det)
--
-- 	    -- move to closest
-- 	    if mv<10 then
-- 	      unclaimedTars[tar]=0
-- 	      unclaimedDets[det]=0
-- -- 	      print('Frame '..(t+1)..'. Moving target update '..tar..' to det '..det)
-- 	      movedState[tar] = det_x[det]:clone()
-- 	    end
--
-- 	  end -- if unclaimedTars
-- 	end -- for tar
--       end -- if ndets
--
--       movedState = movedState:reshape(opt.mini_batch_size,xSize)
--       predictions[t][opt.updIndex] = movedState:clone()
--     end
--     ------------------------------------
--
--     return predictions
-- end


function noisyDA(state, lab, det, allDistBatches)
  state = state:reshape(miniBatchSize*maxTargets, stateDim)
  lab = lab:reshape(miniBatchSize*maxTargets, 1)
  det = det:reshape(miniBatchSize*maxDets, stateDim)
  allDistBatches = allDistBatches:reshape(miniBatchSize,maxTargets,maxDets)

  local maxDist = 0.03
  local scfac = 0.1
  allDistBatches = allDistBatches:cat(torch.FloatTensor(miniBatchSize,maxTargets,1):fill(maxDist),3)

  local newLab = torch.ones(1, nClasses):float()

  --   if miniBatchSize>1 then error('mb') end
  for mb = 1,opt.mini_batch_size do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb

    --     local mbStartD = opt.max_m * (mb-1)+1
    --     local mbEndD =   opt.max_m * mb

    local thisLab = lab[{{mbStart, mbEnd}}]

    --     print(locDist)
    --     print(thisLab)


    --     print(mv,mi)
    for tar=1,maxTargets do
      --       print(tar)
      local locDist = allDistBatches[{{mb},{tar},{}}]:reshape(1,nClasses)
      --       print(locDist)
      local l = thisLab[tar][1]
      local manipulatedDist = locDist:clone()
      --       print(l)
      manipulatedDist[1][l] = 1e2
      --       locDist[1][tar]=1e2
      --       print(locDist)

      local srt, srti = torch.sort(manipulatedDist,2)
      --       print(srt,srti)
      local ratioToClosest = srt[1][1] / locDist[1][l]
      --       print(ratioToClosest)
      local primProb = math.min(.98, scfac*math.exp(ratioToClosest)+(.5-scfac))
      --       primProb = 1 - primProb*torch.rand(1):squeeze()
      --       print(primProb)
      local otherProbs = torch.exp(-locDist)
      otherProbs[1][l] = 0
      local otherProbs = otherProbs / torch.sum(otherProbs,2):squeeze() * (1-primProb)

      local probLine = torch.FloatTensor(1,nClasses)
      for c=1,nClasses do
        if c == l then probLine[1][c] = primProb
        else probLine[1][c] = otherProbs[1][c]
        end
      end
      --       print(otherProbs)
      --
      --       print(probLine)
      --       print(torch.sum(probLine))
      --       abort()
      newLab = newLab:cat(probLine, 1)
    end

    --     print(thisLab)
    --     abort()


  end
  newLab = newLab:sub(2,-1)
  --   print(newLab)
  --   abort()

  --   print(state)
  --   print(lab)
  --   print(det)
  --   print(allDistBatches)
  --   abort()

  return newLab
end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t		time step
-- @param rnn_state	hidden state of RNN (use t-1)
-- @param predictions	current predictions to use for feed back
function getRNNInput(t, rnn_state, predictions, statePred, statePredEx)

  -- make vector statePred into a NxD matrix (maxTargets, stateDim)
  if statePred ~= nil then statePred = statePred:reshape(maxTargets, stateDim) end

  local locdet = detections:clone()
  local input, lab, ex = {}, {}, {}
  local stateLocs, stateLocs2, stateDA, stateEx = {}

  -----  CURRENT  STATE  X(t)  ----
  -- use ground truth input (for training only)
  if opt.use_gt_input ~= 0 then -- use ground truth as input for state
    stateLocs = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize)
    if TESTING then print('USE GT INPUT!!!!!!!!!!!!!!!!!!!!!!!')		 end -- validation and test use previous prediction    
  else
    -- if we are in first frame, take detections as states
    if t==1 then
      local useDet = locdet[{{},{t}}]:clone():reshape(miniBatchSize*maxDets, stateDim)
      local trueDet = detexlabels[{{},{t}}]:eq(1):expand(miniBatchSize*maxDets, stateDim)

      -- useDet = useDet:cmul(trueDet:float())
      useDet = useDet:cmul(dataToGPU(trueDet:float()))
      stateLocs = useDet:clone()
      -- TODO. Make more efficient through :index(...)
      if miniBatchSize>1 then
        --stateLocs = torch.zeros(1,xSize):float()
        stateLocs = zeroTensor2(1,xSize)

        for mb = 1,opt.mini_batch_size do
          local mbStart = opt.max_n * (mb-1)+1
          local mbEnd =   opt.max_n * mb
          local data_batch = locdet[{{mbStart, mbEnd},{t+1}}]:reshape(1,xSize)
          stateLocs = stateLocs:cat(data_batch,1)
        end
        stateLocs=stateLocs:sub(2,-1)
      else -- if not in minibatch mode, just take the first N detections
        stateLocs = stateLocs:sub(1,maxTargets)
      end
    else
      -- for subsequent frames, take previous state as input
      stateLocs, stateLocs2, stateDA, stateEx = decode(predictions,t-1)
    end
  end

  -- override first frame during training with GT
  local inputState = stateLocs:clone():reshape(opt.mini_batch_size,xSize)
  
  if t==1 and TRAINING then
    inputState = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize)
    --     print(inputState)
  end

  local rnninp = {}
  table.insert(rnninp,inputState)


  -----  CURRENT  HIDDEN  STATE  h(t)  ----
  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end

  -----  NEXT  FRAME  DETECTIONS D(t+1)  ----
  local det_x = detections[{{},{t+1}}]:reshape(opt.mini_batch_size, dSize):clone()
  -- wait, do not insert yet... see below

  -----  NEXT  FRAME  DA L(t+1)  ----
  local lab = {}
  -- for training, use ground truth of synthetic data 
  if TRAINING then
    -- TODO: move this into pre-processing for speedup
--    local noisyGT = true
     local noisyGT = false

    -- we want to simulate noise in data association
    if noisyGT then
      local allDistBatches = getPWD(opt, tracks[{{},{t+1}}]:reshape(opt.mini_batch_size,xSize) , detections, t)
      lab = noisyDA(tracks[{{},{t+1}}], labels:narrow(2,t+1,1), detections:narrow(2,t+1,1), allDistBatches)   
    else   
      local detIDforUpdate = labels:narrow(2,t+1,1):long() -- GT labels
      lab = torch.zeros(maxTargets*miniBatchSize,nClasses):float()
      
      for i=1,maxTargets*miniBatchSize do 
        lab[i][detIDforUpdate[i]:squeeze()] = 1
        lab[i] = makePseudoProb(lab[i]:reshape(nClasses),0.0001)
      end
      lab = torch.log(lab)
       
    end
  else
    -- HUNGARIAN
    -- TODO: FIX THIS  
    if opt.use_da_input ~= 0 then error('not implemented anything but HA') end

    local missThr = 0.1; 
    local distNorm = 2; -- Euclidean
    local distFac = 1
    local tarPred = detections[{{},{1}}]:clone():reshape(maxDets,stateDim)
    if t>=2 then tarPred = stateLocs:reshape(maxTargets, stateDim) end    
    local allDist = torch.rand(maxTargets,nClasses):float()
  
    -- TODO: outsource distance computations  
    for tar=1,maxTargets do
      for det=1,maxDets do
        local dist = missThr*2
        --
        --  if det_x[det][1] ~= 0  then
        if detexlabels[det][t+1]==1 then
          dist = torch.dist(tarPred[tar], det_x[det], distNorm)
        end

        allDist[tar][det] = dist
      end
      allDist[tar][nClasses] = missThr
    end

    if maxTargets>1 then
      allDist=allDist:cat(torch.FloatTensor(maxTargets,maxTargets-1):fill(missThr))
    end
           
    local ass = hungarianL(allDist) -- get assignment using HA
    lab = torch.zeros(maxTargets*miniBatchSize,nClasses):float()
    for tar=1,maxTargets do
      local assClass = ass[tar][2]
      if assClass>nClasses then assClass=nClasses end
      lab[tar][assClass] = 1
      lab[tar] = makePseudoProb(lab[tar]:reshape(nClasses):float())
    end    
    lab=torch.log(lab)         
  end
  -- now insert next detections D(t+1)
  table.insert(rnninp,det_x)
  
  -- ... and next data association L(t+1)
  globalLAB = lab:clone()
  lab=lab:reshape(miniBatchSize, maxTargets*nClasses)
  lab = dataToGPU(lab)
  table.insert(rnninp, lab)  
  
  
  
  -----  CURRENT   EXISTENCE   E(t)  ----
  local _,exlab = {},{}  
  if t==1 then 
    exlab = torch.ones(miniBatchSize, maxTargets):float() * 0.5 -- fill all with 1/2 for first frame 
  else  
    _, _, _, exlab=decode(predictions,t-1)        -- get previous prediction for subsequent frames
    exlab=exlab:reshape(miniBatchSize, maxTargets)      
  end
  globalEXLAB = exlab:clone()
  exlab = dataToGPU(exlab)
  table.insert(rnninp, exlab)  
   

  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions	current predictions to use for feed back
-- @param t		time step (nil to predict for entire sequence)
function decode(predictions, t)
  local loctimer=torch.Timer()
  local T = tabLen(predictions)	-- how many frames
  --   local stateLocs = torch.zeros(miniBatchSize* maxTargets,T,fullStateDim)
  --   local stateLocs = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
  --   local stateDA = torch.zeros(miniBatchSize*maxTargets,T,maxTargets)
  --   local stateDA = zeroTensor3(miniBatchSize*maxTargets,T,maxDets)
  --   local stateEx = zeroTensor3(miniBatchSize*maxTargets,T,2)
  --   local nClasses = maxDets
--  if exVar then stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1) end


  local stateLocs, stateLocs2, stateDA, stateEx, stateEx2, stateDynSmooth = {}, {}, {}, {}, {}, {}
  if t ~= nil then
    -- get only one prediction
    local lst = predictions[t] -- list coming from rnn:forward

    if updLoss then stateLocs = lst[opt.updIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if predLoss then stateLocs2 = lst[opt.predIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if daLoss then
      stateDA = lst[opt.daPredIndex]:reshape(miniBatchSize*maxTargets, nClasses) -- miniBatchSize*maxTargets x maxDets
    end
    if exVar then
      stateEx = lst[opt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x 1
    end
    if smoothVar then stateEx2 = lst[opt.exSmoothPredIndex]:reshape(miniBatchSize*maxTargets, 1) end -- miniBatchSize*maxTargets x 1
    if dynSmooth then
    --       stateDynSmooth = lst[opt.dynSmoothPredIndex]:reshape(miniBatchSize,fullStateDim)
    end
  else
    stateLocs = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    stateLocs2 = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    stateDA = zeroTensor3(miniBatchSize*maxTargets,T,nClasses)
    stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1)
    stateEx2 = zeroTensor3(miniBatchSize*maxTargets,T,1)
    stateDynSmooth = zeroTensor3(miniBatchSize*maxTargets,T,fullStateDim)

    for tt=1,T do
      local lst = predictions[tt] -- list coming from rnn:forward
      if updLoss then
        stateLocs[{{},{tt},{}}] = lst[opt.updIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      if predLoss then
        stateLocs2[{{},{tt},{}}] = lst[opt.predIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      --       print(lst)
      --       abort()
      if daLoss then stateDA[{{},{tt},{}}] = lst[opt.daPredIndex]:reshape(miniBatchSize*maxTargets, 1, nClasses) end
      if exVar then
        stateEx[{{},{tt},{}}] = lst[opt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1, 1)
      end
      if smoothVar then
        stateEx2[{{},{tt},{}}] = lst[opt.exSmoothPredIndex]:reshape(miniBatchSize*maxTargets, 1, 1)
      end
      if dynSmooth then
      -- 	stateDynSmooth[{{},{tt},{}}] = lst[opt.dynSmoothPredIndex]:reshape(miniBatchSize*maxTargets, 1, fullStateDim)
      end
    end
  end

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end

  return stateLocs, stateLocs2, stateDA, stateEx, stateEx2, stateDynSmooth
end


--- DA STUFF
--------------------------------------------------------------------------
--- each layer has the same hidden input size
function getDAInitState(opt, miniBatchSize)
  local init_state = {}
  for L=1,DAopt.num_layers do
    local h_init = torch.zeros(miniBatchSize, DAopt.rnn_size)
    table.insert(init_state, dataToGPU(h_init:clone()))
    if DAopt.model == 'lstm' then
      table.insert(init_state, dataToGPU(h_init:clone()))
    end
  end
  return init_state
end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t		time step
-- @param rnn_state	hidden state of RNN (use t-1)
-- @param predictions	current predictions to use for feed back
function getDARNNInput(t, rnn_state, predictions, statePred)
  local loctimer=torch.Timer()

  --- PAIRWISE  DISTANCES ---
  local input = statePred:reshape(opt.mini_batch_size,xSize)
  --   local allDist = getPWD(opt, input, detections, t, detexlabels)

  local allDistBatches = getPWD(opt, input, detections, t, detexlabels)

  --   print(t+1)
  --   print(allDist:reshape(maxTargets,maxDets,stateDim))




  local rnninp = {}
  table.insert(rnninp,allDistBatches)

  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end


  if DAopt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions	current predictions to use for feed back
-- @param t		time step (nil to predict for entire sequence)
function DAdecode(predictions, t)
  local loctimer=torch.Timer()
  local T = tabLen(predictions)	-- how many frames


  local stateLocs, stateLocs2, stateDA, stateEx, DAsum, DAsum2 = {}, {}, {}, {}, {}, {}
  local smoothnessEx = {}
  if t ~= nil then
    -- get only one prediction
    local lst = predictions[t] -- list coming from rnn:forward

    if DAupdLoss then stateLocs = lst[DAopt.updIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if DApredLoss then stateLocs2 = lst[DAopt.predIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if DAdaLoss or true then
      stateDA = lst[DAopt.daPredIndex]:reshape(miniBatchSize, maxTargets * nClasses) -- miniBatchSize*maxTargets x maxDets
      if DAopt.bce ~=0 then
        DAsum = lst[DAopt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
        DAsum2 = lst[DAopt.daSum2PredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
      end
    end
    if DAexVar then
      stateEx = lst[DAopt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x 1
    end
  else
    stateLocs = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    stateLocs2 = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    stateDA = zeroTensor3(miniBatchSize*maxTargets,T,nClasses)
    DAsum = zeroTensor3(miniBatchSize*maxTargets,T,1)
    DAsum2 = zeroTensor3(miniBatchSize*maxTargets,T,1)
    stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1)
    smoothnessEx = zeroTensor3(miniBatchSize*maxTargets,T,1)
    for tt=1,T do
      local lst = predictions[tt] -- list coming from rnn:forward
      if DAupdLoss then
        stateLocs[{{},{tt},{}}] = lst[DAopt.updIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      if DApredLoss then
        stateLocs2[{{},{tt},{}}] = lst[DAopt.predIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      --       print(lst)
      --       abort()
      if DAdaLoss then
        stateDA[{{},{tt},{}}] = lst[DAopt.daPredIndex]:reshape(miniBatchSize*maxTargets, 1, nClasses)
        if DAopt.bce ~=0 then
          DAsum[{{},{tt},{}}] = lst[DAopt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
          DAsum2[{{},{tt},{}}] = lst[DAopt.daSum2PredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
        end
      end
      if DAexVar then
        stateEx[{{},{tt},{}}] = lst[DAopt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1, 1)
        smoothnessEx[{{},{tt},{}}] = lst[DAopt.exPredIndex+1]:reshape(miniBatchSize*maxTargets, 1, 1)
      end
    end
  end


  if DAopt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end

  return stateLocs, stateLocs2, stateDA, DAsum, DAsum2, stateEx, smoothnessEx
end
