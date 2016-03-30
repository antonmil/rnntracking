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
  --   print(costs)
  --   print(maxAllTargets, nClassesHUN, pwdDimHUN)
  local loccost = costs:clone():reshape(miniBatchSize, maxAllTargets*nClassesHUN*pwdDimHUN)
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
    local allstateLocs, allstateLocs2, allstateDA, allexlab=decode(predictions,t)
    -- print(AllstateLocs)
    --       print(stateDA)
    --       abort()
    local stateLocs = torch.zeros(1, stateDim)
    local exlab = torch.zeros(1,1)
    for tar=1,maxAllTargets do
      stateLocs = stateLocs:cat(allstateLocs[tar], 1)
      exlab = exlab:cat(allexlab[tar],1)
    end
    stateLocs = stateLocs:sub(2,-1)
    exlab = exlab:sub(2, -1)

    --       print(stateLocs)
    --       print(exlab)
    --       abort()


    movedState = stateLocs:clone():reshape(maxAllTargets, stateDim)
    local det_x =	alldetections[{{},{t+1}}]:clone():reshape(maxAllDets,stateDim)




    exlab=exlab:reshape(maxAllTargets)
    stateDA = torch.exp(globalLAB)
    --       print(stateDA)
    --       abort()

    local unclaimedDets = torch.ByteTensor(maxAllDets):fill(0)
    local unclaimedTars = torch.ByteTensor(maxAllTargets):fill(0)
    --       print(unclaimedDets)
    for det=1,maxAllDets do
      if torch.max(stateDA:narrow(2,det,1)) < 0.2 and alldetexlabels[det][t+1]==1 then
        unclaimedDets[det]=1
      end
    end
    for tar=1,maxAllTargets do
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
      for tar=1,maxAllTargets do
        if unclaimedTars[tar] == 1 then

          -- get all distances from target to dets
          local allDist = torch.ones(maxAllDets) * 10
          for det=1,maxAllDets do
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

    --       movedState = movedState:reshape(opt.mini_batch_size,maxAllTargets*stateDim)
    movedState = movedState:reshape(maxAllTargets,stateDim)
    --       abort()

    for tar=1,maxAllTargets do
      predictions[tar][t][opt.statePredIndex] = movedState[tar]:clone()
    end
  end
  ------------------------------------

  return predictions
end



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
function getRNNInput(t, Allrnn_states, Allpredictions, statePred, statePredEx)
  local loctimer=torch.Timer()

  -- make vector statePred into a NxD matrix (maxTargets, stateDim)
  if statePred ~= nil then statePred = statePred:reshape(maxTargets, stateDim) end

  local locdet = detections:clone()
  local input, lab, ex = {}, {}, {}
  local stateLocs, stateLocs2, stateDA, stateEx = {}
  local AllstateLocs = {}

  local loctimer2=torch.Timer()
  -- use ground truth input (for training only)
  if opt.use_gt_input ~= 0 then
    stateLocs = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize)
    if TESTING then print('USE GT INPUT!!!!!!!!!!!!!!!!!!!!!!!')		 end
    -- validation and test use previous prediction
  else
    if t==1 then
      local useDet = locdet[{{},{t}}]:clone():reshape(miniBatchSize*maxDets, stateDim)
      local trueDet = detexlabels[{{},{t}}]:eq(1):expand(miniBatchSize*maxDets,stateDim)
      --       print(trueDet)
      --       print(useDet)
      --       abort()
      useDet = useDet:cmul(trueDet:float())
      stateLocs = useDet:clone()

      stateLocs = stateLocs:sub(1,maxTargets)
      for tar=1,maxAllTargets do
        table.insert(AllstateLocs, alldetections[{{tar},{t}}]:reshape(maxTargets, stateDim))
      end
      --       print(allStateLocs)
      --       abort()


      --       print(useDet)
      --       abort()

    else
      AllstateLocs, stateLocs2, stateDA, stateEx = decode(Allpredictions,t-1)
    end
  end

  -- override first frame during training with GT
  --   local inputState = stateLocs:clone():reshape(opt.mini_batch_size,xSize)
  --   if t==1 and TRAINING then
  --     inputState = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize)
  -- --     print(inputState)
  --   end

  local allInputStates = {}
  for tar=1,maxAllTargets do
    table.insert(allInputStates, AllstateLocs[tar]:reshape(opt.mini_batch_size, xSize))
  end
  --   print(allInputStates)
  --   abort()




  local Allrnninps = {}
  for tar=1,maxAllTargets do Allrnninps[tar] = {} end
  for tar=1,maxAllTargets do
    table.insert(Allrnninps[tar],allInputStates[tar])
  end



  local loctimer2=torch.Timer()
  -----  CURRENT   HIDDEN  STATE  h(t)  ----
  for tar=1,maxAllTargets do
    local rnn_state = Allrnn_states[tar]
    for i = 1,#rnn_state[t-1] do table.insert(Allrnninps[tar],rnn_state[t-1][i]) end
  end

  --   print(Allrnninps)
  --   abort()

  local loctimer2=torch.Timer()
  -----  NEXT  FRAME  DETECTIONS D(t+1)  ----
  local det_x = detections[{{},{t+1}}]:reshape(opt.mini_batch_size, dSize):clone()



  -----  NEXT  FRAME  DA L(t+1)  ---- WARNING EXPERIMENT
  local lab = torch.zeros(miniBatchSize * maxTargets, nClasses):float()
  local missThr = 0.1;
  local distNorm = 2; -- Euclidean
  local distFac = 1

  -- training is with ground truth
  if opt.use_da_input==1 then -- WARNING
    if not TRAINING then print('WARNING! using GT not during training!!! t='..t) end

    local noisyGT = true
    --     local noisyGT = false
    if noisyGT then
      local allDistBatches = getPWD(opt, tracks[{{},{t+1}}]:reshape(opt.mini_batch_size,xSize) , detections, t)
      lab = noisyDA(tracks[{{},{t+1}}], labels:narrow(2,t+1,1), detections:narrow(2,t+1,1), allDistBatches)
      --       lab = torch.log(lab)
    else
      detIDforUpdate = labels:narrow(2,t+1,1):long() -- GT labels
      --     print(detIDforUpdate)

      --     print('training')
      lab = torch.zeros(maxTargets*miniBatchSize,nClasses):float()

      for i=1,maxTargets*miniBatchSize do
        lab[i][detIDforUpdate[i]:squeeze()] = 1
        lab[i] = makePseudoProb(lab[i]:reshape(nClasses),0.0001)
      end
    end
    lab = torch.log(lab)
    --     if opt.profiler ~= 0 then  profUpdate('log', loctimer4:time().real) end

  elseif opt.use_da_input==2 then
    lab = DAtmpDA:clone()
  elseif opt.use_da_input==4 or opt.use_da_input==3 then -- HA with ALL detections
    local ifty = 100
    local tarPred = alldetections[{{1,maxAllTargets},{1}}]:clone():reshape(maxAllTargets,stateDim)
    if t>=2 then
      tarPred = torch.zeros(maxAllTargets, stateDim)
      --       tarPred = stateLocs:reshape(maxTargets, stateDim)
      for tar=1,maxAllTargets do
        tarPred[tar] = AllstateLocs[tar]:reshape(1, stateDim)
      end
    end


    local alldet_x =	alldetections[{{},{t+1}}]:clone():reshape(maxAllDets,stateDim)
    --     print('for frame ' .. (t+1))
    --     print('prediction and dets:')
    --
    --     print(tarPred)
    --     print(alldet_x)
    local alldetexvec = alldetexlabels[{{},{t+1}}]:reshape(maxAllDets)
    --     print(alldetexvec:reshape(1,maxAllDets))
    local maxDetsT = torch.sum(alldetexvec)
    --     print(maxDetsT)

    --     local allDist = torch.ones(maxTargets,maxAllDets) * missThr

    local allDist = torch.ones(maxTargets,maxAllDets) * ifty
    for tar=1,maxTargets do
      for det=1,maxAllDets do
        local dist = ifty
        --
        -- 	if det_x[det][1] ~= 0  then
        if alldetexlabels[det][t+1]==1 then
          dist = torch.dist(tarPred[tar], alldet_x[det], distNorm)
        end

        allDist[tar][det] = dist
      end
      --       allDist[tar][nClasses] = missThr
    end

    local fullAllDist = torch.ones(maxAllTargets, maxAllDets+maxAllTargets) * missThr
    for tar=1,maxAllTargets do
      for det=1,maxAllDets do
        if alldetexlabels[det][t+1]==1 then
          dist = torch.dist(tarPred[tar], alldet_x[det], distNorm)
          fullAllDist[tar][det] = dist
        end
      end
    end

    --resolve

    if opt.use_da_input == 4 then
      ass = hungarianL(fullAllDist)


      -- fill in correct detections an L
      local newDetX = alldetections[{{1,maxAllTargets},{t+1}}]:reshape(maxAllTargets,stateDim)
      lab=torch.zeros(maxAllTargets, 2)
      lab:fill(0)
      for tar=1,maxAllTargets do
        local assClass = ass[tar][2]
        if assClass>maxDetsT then
          assClass=nClasses
        else
          newDetX[tar] = alldet_x[assClass]:clone()
          assClass = 1
        end
        lab[tar][assClass] = 1
        lab[tar] = makePseudoProb(lab[tar]:reshape(nClasses):float())
      end
      det_x = newDetX:clone()


    elseif opt.use_da_input == 3 then
      --       local detectionsHUN = detections[{{},{t+1}}]:narrow(3,1,HUNopt.state_dim):reshape(maxDets*miniBatchSize,HUNopt.state_dim)
      --       local tracksHUN = detections[{{},{1}}]:narrow(3,1,HUNopt.state_dim):reshape(maxDets*miniBatchSize,HUNopt.state_dim)
      --       if t>=2 then tarPred = stateLocs:reshape(maxTargets, stateDim) end

      --       costs = getPWDHUN(HUNopt.pwd_mode, nClassesHUN, pwdDimHUN, tracksHUN, detectionsHUN,
      -- 	HUNopt.miss_thr, HUNopt.dummy_noise)
      costs = fullAllDist:clone()

      -- 	print(allDist)
      -- 	print(costs:reshape(maxTargets, nClassesHUN))
      -- 	abort()


      init_stateHUN = getInitStateHUN(HUNopt, miniBatchSize)
      local initStateGlobalHUN = clone_list(init_stateHUN)
      local rnn_stateHUN = {[0] = initStateGlobalHUN}
      --   local predictions = {[0] = {[opt.statePredIndex] = detections[{{},{t}}]}}
      local predictionsHUN = {}
      local DA = {}
      for ttt=1,maxAllTargets do
        local rnninpHUN, rnn_stateHUN = getRNNInputHUN(ttt, rnn_stateHUN, predictionsHUN)		-- get combined RNN input table
        --       print(rnninpHUN)
        local lst = protosHUN.rnn:forward(rnninpHUN)	-- do one forward tick
        predictionsHUN[ttt] = {}
        for k,v in pairs(lst) do predictionsHUN[ttt][k] = v:clone() end -- deep copy

        rnn_stateHUN[ttt] = {}
        for i=1,#init_stateHUN do table.insert(rnn_stateHUN[ttt], lst[i]) end -- extract the state, without output
        DA[ttt] = decodeHUN(predictionsHUN, ttt)

      end
      --     print(predictionsHUN)
      local DAHUN = decodeHUN(predictionsHUN):reshape(maxAllTargets,nClassesHUN)
      --       print(costs)
      --       print(torch.exp(DAHUN))
      --       abort()


       ass=torch.ones(maxAllTargets,2)
      lab = torch.zeros(maxAllTargets,nClasses)

      if sopt.maxProb~=0 then
        -- assignments (max)
        local labprobs = torch.exp(DAHUN)
        local _,maxprobs = torch.max(labprobs,2)
        maxprobs = maxprobs:reshape(maxAllTargets)

        ass = torch.ones(maxAllTargets,2):int()
        for tar=1,maxAllTargets do
          local assClass = maxprobs[tar]
          ass[tar][1] = tar
          ass[tar][2] = assClass
        end

        if sopt.resHA ~=0 then
          ass = hungarianL(-DAHUN)
        end
        -- 	print(ass)

        --       print(det_x:reshape(maxDets, stateDim))
        --       local newDetX = torch.rand(maxDets, stateDim)-.5
        local newDetX = alldetections[{{1,maxAllTargets},{t+1}}]:reshape(maxAllTargets,stateDim)
        lab=torch.zeros(maxAllTargets, 2)
        lab:fill(0)
        for tar=1,maxAllTargets do
          local assClass = ass[tar][2]
          if assClass>maxDetsT then
            assClass=nClasses
          else
            newDetX[tar] = alldet_x[assClass]:clone()
            assClass = 1
          end
          lab[tar][assClass] = 1
          lab[tar] = makePseudoProb(lab[tar]:reshape(nClasses):float())
        end
        --       alldetections[{{},{t+1}}] = newDetX:reshape(maxDets, 1, stateDim)
        --       det_x = detections[{{},{t+1}}]:reshape(opt.mini_batch_size, dSize):clone()
        det_x = newDetX:clone()

      else
        local newDetX = alldetections[{{1,maxAllTargets},{t+1}}]:reshape(maxAllTargets,stateDim)

        -- 	error('how do we interpret smooth probs?')
        local labprobs = torch.exp(DAHUN)
        -- 	print(labprobs)
        -- 	abort()
        -- 	print(maxDetsT)
        lab=torch.zeros(maxAllTargets,2)
        for tar=1,maxAllTargets do
          lab[tar][1] = torch.sum(labprobs[{{tar},{1,maxDetsT}}])
          lab[tar][2] = torch.sum(labprobs[{{tar},{maxDetsT+1,maxAllDets+maxAllTargets}}])
          for d=1,stateDim do
            newDetX[tar][d] = labprobs[{{tar},{1,maxDetsT}}]:reshape(1,maxDetsT) * alldetections[{{1,maxDetsT},{t+1},{1}}]:reshape(maxDetsT,1)
          end
        end
        -- 	print(newDetX)
        det_x = newDetX:clone()
        -- 	print(lab)

      end

    end

    --       print(ass)
    --       abort()




    --       print(newDetX)
    --       print(lab)
    --       sleep(1)

    lab=torch.log(lab)

    globlab = torch.zeros(maxAllTargets, maxAllDets+1)
    for tar=1,maxAllTargets do
      local assClass = ass[tar][2]
      if assClass>maxAllDets then
        assClass=maxAllDets+1
      end
      globlab[tar][assClass] = 1
      globlab[tar] = makePseudoProb(globlab[tar]:reshape(maxAllDets+1):float())
    end
    globlab = torch.log(globlab)

  else
    -- LL based on distance
    local takeAllDets = true
    local takeAllDets = false

    local loctimer3=torch.Timer()
    local allDist = torch.rand(maxTargets,nClasses):float()
    det_x =	detections[{{},{t+1}}]:clone():reshape(maxDets,stateDim)
    local alldet_x = {}

    if takeAllDets then
      alldet_x =	alldetections[{{},{t+1}}]:clone():reshape(maxAllDets,stateDim)
    end
    local tarPred = detections[{{},{1}}]:clone():reshape(maxDets,stateDim)
    if t>=2 then tarPred = stateLocs:reshape(maxTargets, stateDim) end

    if getDetTrick then
      local pwdDim = math.min(2,stateDim)
      local inpPred = tarPred:clone():narrow(2,1,pwdDim)

      local maxDist = 0.05
      --       local inpPred = statePred:reshape(maxTargets,stateDim):narrow(2,1,pwdDim)
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
      --       print(allDist)
      local mv, mi = torch.min(allDist,2) mv = mv:reshape(maxTargets) mi = mi:reshape(maxTargets)
      --       print(mv)

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



    if opt.profiler ~= 0 then  profUpdate('getRNNInput DA', loctimer2:time().real) end
    local loctimer2=torch.Timer()



    --
    local loctimer3=torch.Timer()
--    print(t+1)
--    print(tarPred)
--    print(det_x)
    for tar=1,maxTargets do
      for det=1,maxDets do
        local dist = missThr*2
        --
        -- 	if det_x[det][1] ~= 0  then
        if detexlabels[det][t+1]==1 then
          dist = torch.dist(tarPred[tar], det_x[det], distNorm)
        end

        allDist[tar][det] = dist
      end
      allDist[tar][nClasses] = missThr
    end
    --     print(allDist)
    --     abort()


    local loctimer3=torch.Timer()
    lab = torch.exp(allDist * (-distFac ))
    for tar=1,maxTargets do
      lab[tar] = lab[tar] / torch.sum(lab[tar])
    end    

    -- HUNGARIAN
    local loctimer3=torch.Timer()
    local allD = allDist:clone()
    if maxTargets>1 then
      allDist=allDist:cat(torch.FloatTensor(maxTargets,maxTargets-1):fill(missThr))
    end

    if opt.use_da_input == 0 then
      local ass = hungarianL(allDist)
      --       print(allDist)
--      print(ass)
      --       abort()



      if opt.profiler ~= 0 then  profUpdate('Hungarian', loctimer3:time().real) end
      --       lab = torch.log(lab)

      lab = torch.zeros(maxAllTargets*miniBatchSize,nClasses):float()
      lab:fill(0)
      for tar=1,maxTargets do
        local assClass = ass[tar][2]
        if assClass>nClasses then assClass=nClasses end
        lab[tar][assClass] = 1
        lab[tar] = makePseudoProb(lab[tar]:reshape(nClasses):float())
      end


      lab=torch.log(lab)
      
      globlab = torch.zeros(maxAllTargets, maxAllDets+1)
      for tar=1,maxAllTargets do
        local assClass = ass[tar][2]
        if assClass>maxAllDets then
          assClass=maxAllDets+1
        end
        globlab[tar][assClass] = 1
        globlab[tar] = makePseudoProb(globlab[tar]:reshape(maxAllDets+1):float())
      end
      globlab = torch.log(globlab)
          


    end

  end


  for tar=1,maxAllTargets do
    table.insert(Allrnninps[tar],det_x[tar]:reshape(1,stateDim))
  end

  --   print(Allrnninps)
  --   print('passing lab')
  --   print(torch.exp(lab))
  --   abort()
--  print(lab)
--  abort()
--globalLAB = lab:clone()
  globalLAB = globlab:clone()
  --   if miniBatchSize>1 then lab=lab:reshape(miniBatchSize, maxTargets*nClasses) end
  --   print(lab)
  --   lab=lab:reshape(miniBatchSize, maxTargets*nClasses)
  --   print(lab)

  --   table.insert(Allrnninps, lab)

  for tar=1,maxAllTargets do
    table.insert(Allrnninps[tar],lab[tar]:reshape(1,nClasses))
  end


  local loctimer2=torch.Timer()
  if opt.einp~=0 then
    local Allexlabs = {}

    if TRAINING and false then -- WARNING
      exlab = exlabels[{{},{t}}]:clone():reshape(miniBatchSize * maxTargets, 1):float()
    else
      if t==1 then
        for tar=1,maxAllTargets do
          table.insert(Allexlabs, torch.ones(miniBatchSize, maxTargets):float() * 0.5)
        end
      else
        _, _, _, Allexlabs=decode(Allpredictions,t-1)


        for tar=1,maxAllTargets do
          Allexlabs[tar]=Allexlabs[tar]:reshape(miniBatchSize, maxTargets)
        end
        -- 	exlab=exlab:reshape(maxTargets)
      end
    end
    --     globalEXLAB = exlab:clone()

    for tar=1,maxAllTargets do
      table.insert(Allrnninps[tar], Allexlabs[tar])
    end
  end





  if opt.profiler ~= 0 then  profUpdate('getRNNInput ExLab', loctimer2:time().real) end

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end
  return Allrnninps, Allrnn_states
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions	current predictions to use for feed back
-- @param t		time step (nil to predict for entire sequence)
function decode(Allpredictions, t)
  local loctimer=torch.Timer()

  AllstateLocs, AllstateLocs2, AllstateDA, AllstateEx, AllstateEx2, AllstateDynSmooth = {}, {}, {}, {}, {}, {}
  for tar=1,maxAllTargets do
    --     print(Allpredictions)
    --     print(Allpredictions[tar])
    local predictions = Allpredictions[tar]
    local T = tabLen(predictions)	-- how many frames
    --   local stateLocs = torch.zeros(miniBatchSize* maxTargets,T,fullStateDim)
    --   local stateLocs = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    --   local stateDA = torch.zeros(miniBatchSize*maxTargets,T,maxTargets)
    --   local stateDA = zeroTensor3(miniBatchSize*maxTargets,T,maxDets)
    --   local stateEx = zeroTensor3(miniBatchSize*maxTargets,T,2)
    --   local nClasses = maxDets
    if exVar then stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1) end


    local stateLocs, stateLocs2, stateDA, stateEx, stateEx2, stateDynSmooth = {}, {}, {}, {}, {}, {}
    if t ~= nil then
      -- get only one prediction
      --       print('aaa')
      --       print(predictions)
      local lst = predictions[t] -- list coming from rnn:forward

      if updLoss then stateLocs = lst[opt.statePredIndex]:clone():reshape(miniBatchSize*maxTargets, fullStateDim) end -- miniBatchSize x maxTargets*fullStateDim
      if predLoss then stateLocs2 = lst[opt.statePredIndex2]:clone():reshape(miniBatchSize*maxTargets, fullStateDim) end -- miniBatchSize x maxTargets*fullStateDim
      if daLoss then
        stateDA = lst[opt.daPredIndex]:clone():reshape(miniBatchSize*maxTargets, nClasses) -- miniBatchSize*maxTargets x maxDets
      end
      if exVar then
        stateEx = lst[opt.exPredIndex]:clone():reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x 1
      end
      if smoothVar then stateEx2 = lst[opt.exSmoothPredIndex]:clone():reshape(miniBatchSize*maxTargets, 1) end -- miniBatchSize*maxTargets x 1
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
          -- 	  print(lst)
          stateLocs[{{},{tt},{}}] = lst[opt.statePredIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
        end
        if predLoss then
          stateLocs2[{{},{tt},{}}] = lst[opt.statePredIndex2]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
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

    --   return stateLocs, stateLocs2, stateDA, stateEx, stateEx2, stateDynSmooth
    table.insert(AllstateLocs, stateLocs:clone())
    table.insert(AllstateLocs2, stateLocs2:clone())
    table.insert(AllstateDA, stateDA)
    table.insert(AllstateEx, stateEx:clone())
    table.insert(AllstateEx2, stateEx2)
    table.insert(AllstateDynSmooth, stateDynSmooth)
  end
  return AllstateLocs, AllstateLocs2, AllstateDA, AllstateEx, AllstateEx2, AllstateDynSmooth
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

  --   local nClasses = maxDets+2
  if exVar then stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1) end


  local stateLocs, stateLocs2, stateDA, stateEx, DAsum, DAsum2 = {}, {}, {}, {}, {}, {}
  local smoothnessEx = {}
  if t ~= nil then
    -- get only one prediction
    local lst = predictions[t] -- list coming from rnn:forward

    if DAupdLoss then stateLocs = lst[DAopt.statePredIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if DApredLoss then stateLocs2 = lst[DAopt.statePredIndex2] end -- miniBatchSize x maxTargets*fullStateDim
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
        stateLocs[{{},{tt},{}}] = lst[DAopt.statePredIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      if DApredLoss then
        stateLocs2[{{},{tt},{}}] = lst[DAopt.statePredIndex2]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
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
