require 'torch'
require 'nn'
require 'nngraph'

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




--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t		time step
-- @param rnn_state	hidden state of RNN (use t-1)
-- @param predictions	current predictions to use for feed back
function getRNNInput(t, rnn_state, predictions)
  local loctimer=torch.Timer()
  
  local locdet = detections:clone()
  local locex = exlabels:clone()
  local loclab = labels:clone()
  local input, lab, ex = {}, {}, {}
  if opt.use_gt_input ~= 0 then 
    input = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize) --WARNING! SHOULDNT BE t+1???!?!
--     print(labels[{{},{t}}])
--     print(opt.mini_batch_size, maxTargets, maxDets)
    
--     print(labOneHot)
--     sleep(2)
--     error('use gt input')
    if opt.linp~=0 then 
--       local labOneHot = getOneHotLab2(labels[{{},{t}}]:reshape(maxTargets*opt.mini_batch_size), miniBatchSize==1)
--       print(labOneHot)
--       abort()
      
      if TRAINING or t==1 then
	
	
	lab = torch.zeros(1,nClasses):float()
	for tar=1,maxTargets do
	  local tmpVec = torch.zeros(nClasses)
	  tmpVec[labels[tar][t]] = 1
	  local labOneHot = makePseudoProb(tmpVec:reshape(nClasses):float())

	  lab = lab:cat(labOneHot:reshape(1,nClasses),1)
	end
	lab=lab:sub(2,-1)
	lab = torch.log(lab)
	
-- 	lab = torch.Tensor(maxTargets):fill(1):float() -- WARNING TERMINATION ONLY
-- 	lab = exlabels[{{},{t}}]:clone():float()
-- 	if t>1 then lab = exlabels[{{},{t-1}}]:clone():float() end
-- 	lab = lab - torch.random(maxTargets,1)*0.001
	
      else
-- 	local _, _, stateDA = decode(predictions,t-1)
-- 	lab = stateDA:clone()
-- 	local _, _, _, stateEx= decode(predictions,t-1)
-- 	lab = stateEx:clone()
-- 	if lab[1][1]>0.5 then lab[1][1]=1 else lab[1][1] = 0 end
-- 	lab = exlabels[{{},{t-1}}]:clone():float()
      end
--       print(TRAINING or t==1)
--       print(lab)
--       if (not (TRAINING or t==1)) then  sleep(1) end
      
    end
  else
    if t==1 then 
    local tt1=torch.Timer()
--       print(detections)    
      local maxD = opt.max_m if opt.max_m > opt.max_n+opt.max_nf then maxD = opt.max_n end
      local oD = torch.zeros(1,1,stateDim):float()
--       local oD = zeroTensor3(1,1,stateDim)
--       print(locdet)
--       oD = dataToGPU(oD)
      for m=1,opt.mini_batch_size do
	local mbOffset=(m-1)*opt.max_m
	oD = oD:cat(locdet[{{1+mbOffset,maxD+mbOffset},{t}}]:float(),1)
-- 	oD = torch.cat(oD, locdet[{{1+mbOffset,maxD+mbOffset},{t}}],1)
      end
      oD=oD:sub(2,-1) -- remove first
      oD = dataToGPU(oD)
      
--       print(oD)
      input =  	oD:reshape(opt.mini_batch_size,xSize)
--       input = detections[{{},{t}}]:reshape(opt.mini_batch_size, xSize)
      
      if stateVel then	
	input = torch.cat(input:float():t(),torch.zeros(opt.mini_batch_size, xSize):float():t(), 2):reshape(opt.mini_batch_size,fullxSize)
	input = dataToGPU(input)
      end
      
      if opt.linp~=0 then lab = getFirstFrameLab(miniBatchSize==1) end
      
      if opt.einp~=0 then 
	if miniBatchSize==1 then ex = firstFrameExT
	else ex = firstFrameEx end
      end
      input = tracks[{{},{t}}]:reshape(opt.mini_batch_size,xSize) 
--       print(lab)
--       abort()
      if opt.profiler ~= 0 then  profUpdate('getRNN 1', tt1:time().real) end 
    else
      local tt1=torch.Timer()
      local stateLocs, stateLocs2, stateDA, stateEx = decode(predictions,t-1)
      input =  	stateLocs:reshape(opt.mini_batch_size,fullxSize)
--       print(stateDA)
--       print(maxTargets, maxDets)
      if opt.linp~=0 then lab =  	stateDA:reshape(opt.mini_batch_size,maxTargets*nClasses) end
--       print(stateEx)
      if opt.einp~=0 then stateEx:reshape(miniBatchSize,1*maxTargets) end
      
      if opt.profiler ~= 0 then  profUpdate('getRNN t', tt1:time().real) end 
      
--       print(lab)
--       abort()
    end
  end
  
  --- PAIRWISE  DISTANCES ---
  local allDistBatches = getPWD(opt, input, detections, t)




  local rnninp = {}
  table.insert(rnninp,allDistBatches)
  
  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end
  
--   print(labels[{{},{t}}])
  
--   if opt.linp~=0 then table.insert(rnninp, lab) end
--   if opt.einp~=0 then table.insert(rnninp, ex) end
--   
--   for B=1,opt.batch_size do
-- --     print(detections)
--     local det_x =		detections[{{},{t+B}}]:reshape(opt.mini_batch_size, dSize):clone()
-- --     det_x = det_x:cat(input:reshape(opt.mini_batch_size, xSize), 2)  
-- 
--     
--     
-- --     local det_x = torch.zeros(opt.mini_batch_size, dSize):float()
-- --     for i=1,opt.max_m do
-- --       det_x[1][i] = torch.norm(detections[{{i},{t+B},{1}}] - input:squeeze())
-- -- --       print(det_x[1][i])
-- --     end
-- --     abort()
--     
--     det_x[det_x:eq(0)]=opt.dummy_det_val
--     table.insert(rnninp,det_x)  
--     
-- --     local det_x_ifty = det_x:clone()
-- --     local ifty = 0
-- --     det_x_ifty[det_x_ifty:eq(0)] = ifty
-- --     table.insert(rnninp,det_x_ifty)    
--   end
  
--   local prevLab = loclab[{{},{t}}]:reshape(opt.mini_batch_size,nClasses)
  
  
  
  
  -- SMOOTHNESS
  if opt.einp~=0 then
    local ex = {}
    if t>1 then
	  local _, _, _, stateEx= decode(predictions,t-1)
	  ex = stateEx:clone()  
    else
      ex = torch.ones(maxTargets):float()    
    end
    
    if TRAINING then
      ex = exlabels[{{},{t}}]:clone():float():reshape(maxTargets)
    end

    
    
      table.insert(rnninp, ex) 
  end
	---
  
  
  
  
--   if opt.linp~=0 then tabl.insert(rnninp, 


  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end 
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
--   local nClasses = maxDets+2
  if exVar then stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1) end


  local stateLocs, stateLocs2, stateDA, stateEx, DAsum, DAsum2 = {}, {}, {}, {}, {}, {}
  local smoothnessEx = {}
  if t ~= nil then
    -- get only one prediction
    local lst = predictions[t] -- list coming from rnn:forward

    if updLoss then stateLocs = lst[opt.statePredIndex] end -- miniBatchSize x maxTargets*fullStateDim
    if predLoss then stateLocs2 = lst[opt.statePredIndex2] end -- miniBatchSize x maxTargets*fullStateDim
    if daLoss then 
      stateDA = lst[opt.daPredIndex]:reshape(miniBatchSize,maxTargets* nClasses) -- miniBatchSize*maxTargets x maxDets
      if opt.bce~=0 then
-- 	DAsum = lst[opt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, nClasses) -- miniBatchSize*maxTargets x maxDets
	DAsum = lst[opt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
	DAsum2= lst[opt.daSum2PredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
      end
--       stateDA = lst[opt.daPredIndex]:reshape(miniBatchSize, maxTargets* nClasses) -- miniBatchSize*maxTargets x maxDets
    end
    if exVar then 
      stateEx = lst[opt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x 1
    end
    if smoothVar then 
      smoothnessEx = lst[opt.exPredIndex+1]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x 1
    end
    
    
  else    
    stateLocs = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)
    stateLocs2 = zeroTensor3(miniBatchSize* maxTargets,T,fullStateDim)  
    stateDA = zeroTensor3(miniBatchSize*maxTargets,T,nClasses)
--     DAsum = zeroTensor3(miniBatchSize*maxTargets,T,nClasses)
    DAsum = zeroTensor3(miniBatchSize*maxTargets,T,1)
    DAsum2 = zeroTensor3(miniBatchSize*maxTargets,T,1)  
    stateEx = zeroTensor3(miniBatchSize*maxTargets,T,1)
    smoothnessEx = zeroTensor3(miniBatchSize*maxTargets,T,1)
    for tt=1,T do
      local lst = predictions[tt] -- list coming from rnn:forward
      if updLoss then 
	stateLocs[{{},{tt},{}}] = lst[opt.statePredIndex]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)
      end
      if predLoss then 
	stateLocs2[{{},{tt},{}}] = lst[opt.statePredIndex2]:clone():reshape(miniBatchSize* maxTargets, 1, fullStateDim)    
      end
--       print(lst)
--       abort()
      if daLoss then 
	stateDA[{{},{tt},{}}] = lst[opt.daPredIndex]:reshape(miniBatchSize*maxTargets, 1, nClasses) 
	if opt.bce~=0 then
-- 	  DAsum[{{},{tt},{}}] = lst[opt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, 1, nClasses) -- miniBatchSize*maxTargets x maxDets
	  DAsum[{{},{tt},{}}] = lst[opt.daSumPredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
	  DAsum2[{{},{tt},{}}] = lst[opt.daSum2PredIndex]:reshape(miniBatchSize*maxTargets, 1) -- miniBatchSize*maxTargets x maxDets
	end
      end
      if exVar then 
	stateEx[{{},{tt},{}}] = lst[opt.exPredIndex]:reshape(miniBatchSize*maxTargets, 1, 1)    
      end
      if smoothVar then 
	smoothnessEx[{{},{tt},{}}] = lst[opt.exPredIndex+1]:reshape(miniBatchSize*maxTargets, 1, 1)    
      end
      
    end
  end
  
  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end 

  return stateLocs, stateLocs2, stateDA, DAsum, DAsum2, stateEx, smoothnessEx
end