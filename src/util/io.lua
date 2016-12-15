--------------------------------------------------------------------------
--- Reads a txt file format from MOTChallenge 2015 (frame, id, bbx,...).
-- @param datafile - path to data file
-- @param mode (optional)  1 = ground truth / result, 2 = detections
-- @return state  - a table with bounding boxes
-- @return data[t][id] can be used to access one specific box
-- @see writeTXT
function readTXT(datafile, mode)
  -- local datafile = '/media/sf_vmex/2DMOT2015/data/TUD-Campus/gt/gt.txt'

  local gtraw = csvRead(datafile)
  local data={}
  local confThr = -1e5
  if opt~= nil and opt.detConfThr ~= nil then confThr = opt.detConfThr end
  if sopt~=nil and sopt.detConfThr ~= nil then confThr = sopt.detConfThr end
--   print(confThr)
    
  
  
  if not mode then
	-- figure out whether we are in GT/Result (1) or in Det (2) mode
	mode = 1
	if gtraw[1][7] ~= -1 then mode = 2 end -- very simple, gt do not have scores
  end
  -- go through all lines
  for l = 1,tabLen(gtraw) do    
    fr=gtraw[l][1]
    id=gtraw[l][2]
    bx=gtraw[l][3]
    by=gtraw[l][4]
    bw=gtraw[l][5]
    bh=gtraw[l][6]
    sc=gtraw[l][7]
    if data[fr] == nil then
      data[fr] = {}      
    end
    -- detections do not have IDs, simply increment
    if mode==2 then id = table.getn(data[fr]) + 1 end
    
    -- only use box for ground truth / result, and box + confidence for detections
    if mode == 1 then
      table.insert(data[fr],id,torch.Tensor({bx,by,bw,bh}):resize(1,4)) 
    elseif mode == 2 and sc > confThr then      
      table.insert(data[fr],id,torch.Tensor({bx,by,bw,bh,sc}):resize(1,5)) 
    end
  end
  
  -- shift (left,top) to (center_x, center_y)  
  for k,v in pairs(data) do
    for l,m in pairs(v) do
      local box = data[k][l]:squeeze()
      box[1]=box[1]+box[3]/2 -- this already changes data in place
      box[2]=box[2]+box[4]/2
--       data[k][l] = box:reshape(1,box:nElement())
    end
  end    

  -- return table
  return data
end


--------------------------------------------------------------------------
--- Write a txt file format of MOTChallenge 2015 (frame, id, bbx,...)
-- @param data	The tensor containing bounding boxes, a FxNxD tensor.
-- @param datafile Path to data file.
-- @param thr 	A threshold for ignoring boxes.
-- @param mode 	(optional)  1 = result, 2 = detections
-- @see readTXT
function writeTXT(data, datafile, thr, mode)
  thr = thr or 0
  mode = mode or 1 	-- defaults to bounding boxes with id
  
  local N,F,D = getDataSize(data)
  local stateDim = math.min(4,D)
  
  -- how many boxes are present
  nBoxes = torch.sum(torch.ne(data:narrow(3,1,1):squeeze(),0))
  nBoxes = 0
  for i=1,N do for t=1,F do
    if torch.sum(torch.abs(data[{{i},{t},{1,stateDim}}])) ~= 0 then nBoxes = nBoxes + 1 end
  end
  end
  
--   print(data)
--   print(nBoxes)
--   abort()
--   
  
  -- Shift cx,cy back to left,top
--   print(data[1])
  if D>=4 then
    for i=1,N do
      for t=1,F do
	data[i][t][1] = data[i][t][1] - data[i][t][3]/2
	data[i][t][2] = data[i][t][2] - data[i][t][4]/2
      end
    end
  end 
--   print(data[1])
  
  
  local out = torch.Tensor(nBoxes, 7):fill(-1)	-- the tensor to be written
--   print(out:size())
--   print(data:size())
  
  if mode==2 then error("writeTXT for detections not implemented") end

  bcnt=0 -- box counter
  for t=1,data:size(2) do    
    for i=1,data:size(1) do
--       x = data[i][t][1] -- x coordinate
--       nz = 0
--       for d=1,D do if data[i][t][d]~=0 t
      if torch.sum(torch.abs(data[{{i},{t},{1,stateDim}}])) ~= 0 then -- if all coordinates 0, ignore
	bcnt=bcnt+1
-- 	print(bcnt)
	out[bcnt][1] = t
	out[bcnt][2] = i
	for d=1,data:size(3) do
	  out[bcnt][d+2] = data[i][t][d]
	end
      end
      
    end
  end

  csvWrite(datafile, out)
  
end



--------------------------------------------------------------------------
--- Construct a model-specific file name
-- @param base 		Model name
-- @param opt		Model options
-- @param modelParams 	A list of model-specific parameters
-- @return 		Full file name
-- @return 		Directory
-- @return 		Base file (model name)
-- @return 		Model-specific part (signature)
-- @return 		File extension
function getCheckptFilename(base, opt, modelParams)
  local tL = tabLen(modelParams)	-- how many relevant parameters
  
  local ext = '.t7'			-- file extension
  local dir = getRNNTrackerRoot()..'bin/'			-- directory  
  local signature = ''
  
  for i=1,tL do
    local p = opt[modelParams[i]]
    local pr = ''			-- prepend suffix
    if modelParams[i] == 'model_index' 		then pr='mt' 
    elseif modelParams[i] == 'rnn_size' 	then pr='r' 
    elseif modelParams[i] == 'num_layers' 	then pr='l' 
    elseif modelParams[i] == 'max_n' 		then pr='n'
    elseif modelParams[i] == 'max_m' 		then pr='m'
    elseif modelParams[i] == 'state_dim' 	then pr='d' 
    elseif modelParams[i] == 'vel'	 	then pr='v'
    elseif modelParams[i] == 'loss_type'	then pr='lt'
    elseif modelParams[i] == 'linp'		then pr='li'
    elseif modelParams[i] == 'batch_size' 	then pr='b'
    elseif modelParams[i] == 'train_type'	then pr='y' end
    signature = signature .. pr
    if p==torch.round(p) then
      signature = signature .. string.format('%d',p)	-- append parameter
    else 
      signature = signature .. string.format('%.2f',p)	-- append parameter
    end
    if i<tL then signature = signature .. '_' end	-- append separator 
  end
  
  fn = dir .. base .. '_' .. signature .. ext			-- append extension
  
  return fn, dir, base, signature, ext
end

-- TODO in progress, unused so far...
function getTrainingDataFilename(opt, dataParams, mode)
  mode = mode or 'train'
  
  local tL = tabLen(dataParams)	-- how many relevant parameters
  
  local ext = '.t7'			-- file extension
  local dir = getRNNTrackerRoot()..'tmp/'			-- directory  
  local signature = ''
  
  for i=1,tL do
    local p = opt[dataParams[i]]
    local pr = ''			-- prepend suffix
    if dataParams[i] == 'synth_training' 	then pr='st'
    elseif dataParams[i] == 'synth_valid' 	then pr='sv'
    elseif dataParams[i] == 'mini_batch_size' 	then pr='mb' 
    elseif dataParams[i] == 'max_n' 		then pr='n'
    elseif dataParams[i] == 'max_m' 		then pr='m'
    elseif dataParams[i] == 'state_dim' 	then pr='d' 
    elseif dataParams[i] == 'vel'	 	then pr='v'
    elseif dataParams[i] == 'full_set'	 	then pr='f'
    elseif dataParams[i] == 'fixed_n'	 	then pr='fn'    
    elseif dataParams[i] == 'temp_win'		then pr='t'
    elseif dataParams[i] == 'real_data'		then pr='rda'
    elseif dataParams[i] == 'real_dets' 	then pr='rde'
    elseif dataParams[i] == 'trim_tracks' 	then pr='tt' 
    end
    signature = signature .. pr
    if p==torch.round(p) then
      signature = signature .. string.format('%d',p)	-- append parameter
    else 
      signature = signature .. string.format('%.2f',p)	-- append parameter
    end
    if i<tL then signature = signature .. '_' end	-- append separator 
  end
  
  fn = dir .. mode .. '_' .. signature .. ext			-- append extension
  
  return fn, dir, mode, signature, ext  
end

--------------------------------------------------------------------------
--- Save checkpoint (convert to CPU first)
function saveCheckpoint(savefile, tracks, detections, protos, opt, trainLosses, time, it)  
  savefile = savefile or string.format('%sbin/model.t7',getRNNTrackerRoot())
  print('saving model to ' .. savefile)
  local checkpoint = {}
  -- checkpoint.detections = detections:float()
  -- checkpoint.gt = tracks:float()
  for k,v in pairs(protos) do protos[k] = protos[k]:float() end
  checkpoint.protos = protos
  checkpoint.opt = opt
  -- checkpoint.trainLosses = trainLosses
  checkpoint.i = opt.max_epochs
  checkpoint.epoch = opt.max_epochs  
  checkpoint.time = time
  checkpoint.it = it
  torch.save(savefile, checkpoint)
end


--------------------------------------------------------------------------
--- Load synthetic data
function loadSynthTraining(opt,mode)
  local valid = true
  print('Can we load training / validation data?')
--   if opt.training_file == nil or opt.training_file == '' then return false end
  if dataParams == nil then 
    print('No. no dataPrams')
    return false 
  end
  
  local fn = getTrainingDataFilename(opt, dataParams, mode)
  if not lfs.attributes(fn,'mode') then 
    print('No. No file available '..fn);
    return false 
  end 		
  local trData = torch.load(fn)
  
  
  local allRelParams  ={}
  for k,v in pairs(dataParams) do table.insert(allRelParams, v) end
  table.insert(allRelParams,'det_noise')
  table.insert(allRelParams,'det_fail')
  table.insert(allRelParams,'det_false')
  table.insert(allRelParams,'norm_std')
  table.insert(allRelParams,'norm_mean')
  table.insert(allRelParams,'dummy_det_val')
  table.insert(allRelParams,'trim_tracks')  
  table.insert(allRelParams,'fixed_n')
--   print('a'
  
  for k,v in pairs(allRelParams) do
    print(v,opt[v], trData.opt[v])
    if opt[v] ~= trData.opt[v] then 
      print('No. opt are different: '..v..opt[v]..trData.opt[v])
      return false 
    end
  end
  
  return valid
end

--------------------------------------------------------------------------
--- Save synthetic data
function saveSynthTraining(opt, TracksTab, DetsTab, LabTab, ExTab, DetExTab, SeqNames, mode)
  
  if dataParams == nil then return false end
--   if lfs.attributes('/home/h3/','mode') then return false end 			-- network
  if mode~='train' and mode~= 'validation' then return false end
    
  local fn = getTrainingDataFilename(opt, dataParams,mode)
  
  local trData = {}
  trData['TracksTab']=TracksTab
  trData['DetsTab']=DetsTab
  trData['LabTab']=LabTab
  trData['ExTab']=ExTab
  trData['DetExTab']=DetExTab
  trData['SeqNames']=SeqNames  
  trData['opt']=opt
  trData['dataParams']=dataParams
  
  pm('Saving training data to '..fn)
  torch.save(fn, trData)
  
end
