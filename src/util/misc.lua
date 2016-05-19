-----------------------
--- Module including many miscellaneous utility functions
--
-- @module types

require 'gnuplot'	-- plotting stuffgetTracksAndDetsTables
require 'lfs' 		-- luaFileSystem, check for dir, etc...
require 'util.csv'	-- reading CSV files
require 'image'		-- image load
require 'util.plot'	-- plotting utils
require 'util.io'	-- reading / writing data
require 'util.matlab'	-- matlab like tools, union, intersect, find,...
require 'util.paths' 	-- paths to data, etc...
require 'util.stdio' 	-- various printing functions
require 'util.data' 	-- data reading, generating, preparing...
require 'external.hungarian'
require 'distributions'

--------------------------------------------------------------------------
--- Utility function. TODO: move away to some utils file?
-- takes a list of tensors and returns a list of cloned tensors
-- @author Andrej Karpathy
function clone_list(tensor_list, zero_too)        
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

--------------------------------------------------------------------------
--- Sleep for n seconds.
-- @param n Time to hold in seconds.
function sleep(n)  -- seconds
  local clock = os.clock
  local t0 = clock()
  while clock() - t0 <= n do end
end

--------------------------------------------------------------------------
--- Abort the execution of the program (for debugging)
-- @param msg 	optional abort message
function abort(msg) 
  msg = msg or ''
  print("Aborting "..msg)
  os.exit(0)
end


--------------------------------------------------------------------------
--- Reshuffle detections to remove IDs
-- @param detections	The data tensor
-- @return scrambled detections and corresponding indices
function reshuffleDets(detections)
  local scrdet = detections:clone()
  local N,F = getDataSize(detections)
  local scrindex = torch.IntTensor(N, F)
  for t=1,F do
    scrindex[{{},{t}}] = torch.randperm(N)
    for i = 1,N do    
      scrdet[i][t] = detections[scrindex[i][t]][t]
    end
  end
  
  return scrdet, scrindex
end

--------------------------------------------------------------------------
--- Reshuffle tracks
-- @param data	The data tensor
function reshuffleTracks(data)
  local scrdata = data:clone()
  local N,F,D = getDataSize(data)
  local scrindex = torch.IntTensor(N)
  scrindex = torch.randperm(N)
  for i=1,N do
    scrdata[{{i},{},{}}] = data[{{scrindex[i]},{},{}}]
  end
  return scrdata, scrindex
end




--------------------------------------------------------------------------
--- Get ground truth of a specific sequence as a table.
-- @param seqName	The name of the sequence.
function getGT(seqName)
  local datafile = getDataDir() .. seqName .. "/gt/gt.txt"
  if lfs.attributes(datafile) then 
--     print("GT file ".. datafile)
  else
    error("Error: GT file ".. datafile .." does not exist")
  end
  return readTXT(datafile, 1) -- param #2 = GT
end

--------------------------------------------------------------------------
--- Get result of a specific sequence as a table.
-- @param seqName	The name of the sequence.
function getRes(seqName, resDir)
  local datafile = resDir .. seqName .. ".txt"
  if lfs.attributes(datafile) then 
--     print("GT file ".. datafile)
  else
    error("Error: Results file ".. datafile .." does not exist")
  end
  return readTXT(datafile, 1) -- param #2 = GT
end


--------------------------------------------------------------------------
--- Get ground truth of a sequence as a tensor. NOTE All tracks are retrieved.
-- @param seqName 	The name of the sequence.
function getGTTracks(seqName)
  ts = ts or 1 	-- default starting frame
  local nDim = 4		-- number of dimensions
  
  local gt = getGT(seqName)	-- get raw GT table
  
 
  local imgWidth = 1
--   F = math.min(tabLen(gt)-ts+1,F)
  local tL,minKey, maxKey = tabLen(gt)
  local F = maxKey
  local N = 255
  
  local tracks = torch.ones(N, F, nDim):fill(0)
  
  -- Now create a FxN tensor with states
  for t in pairs(gt) do
    local cnt=0
    for i = 1, N do
      cnt=cnt+1
      -- 	local gtID, gtBBox = orderedKeys[i], gt[t+ts-1][ orderedKeys[i] ]
      local gtBBox = gt[t][i]
      if gtBBox then 
	for dim = 1,nDim do
      -- 	    tracks[cnt][t] = getFoot(gtBBox)[1][1] / imgWidth 	    
	      tracks[cnt][t][dim]=gtBBox[1][dim] / imgWidth 	-- normalize to [0,1]
	end
      end
    end
  end
  
  -- remove zero rows
  tracks = cleanDataTensor(tracks)
  return tracks
end

--------------------------------------------------------------------------
--- TODO
-- @param fileName 
function getData(fileName, mode)
  local nDim = 4		-- number of dimensions
  
  local raw = csvRead(fileName)
  local data={}
  
  if not mode then
	-- figure out whether we are in GT (1) or in Det (2) mode
	mode = 1
	if raw[1][7] ~= -1 then mode = 2 end -- very simple, gt do not have scores
  end
  
  
  -- go through all lines
  for l = 1,tabLen(raw) do    
    local fr=raw[l][1]
    local id=raw[l][2]
    local bx=raw[l][3]
    local by=raw[l][4]
    local bw=raw[l][5]
    local bh=raw[l][6]
    local sc=raw[l][7]
    if data[fr] == nil then
      data[fr] = {}      
    end
    -- detections do not have IDs, simply increment
    if mode==2 then id = table.getn(data[fr]) + 1 end
    
    -- only use box for ground truth, and box + confidence for detections
    table.insert(data[fr],id,torch.Tensor({bx,by,bw,bh,sc}):resize(1,5)) 
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

  -- all info in data table
  local tL,minKey, maxKey = tabLen(data)
  local F = maxKey
  local N = 50
  local tracks = torch.ones(N, F, nDim):fill(0)
  local labs = torch.IntTensor(N, F):fill(0)
  
  -- Now create a FxN tensor with states
  for t in pairs(data) do
    local cnt=0
    for i = 1, N do
      cnt=cnt+1
      -- 	local gtID, gtBBox = orderedKeys[i], gt[t+ts-1][ orderedKeys[i] ]
      local gtBBox = data[t][i]
--       print(t,i, gtBBox)
--       abort()
      if gtBBox then 
	for dim = 1,nDim do
	  tracks[cnt][t][dim]=gtBBox[1][dim]	    
	end
	labs[cnt][t] = gtBBox[1][5]
      end
    end
  end
--   print(tracks)
--   abort()
  
  
  return tracks, labs
  

end

-- function getDetLabels(gtfile)
--   
--   gtRaw = csvRead(gtfile)
--   local data={}
-- --   print(detRaw)
--   
--   -- go through all lines
--   for l = 1,tabLen(gtRaw) do    
--     local fr=gtRaw[l][1]
--     local id=gtRaw[l][2]
--     local da=gtRaw[l][7]
--     if data[fr] == nil then
--       data[fr] = {}
--       
--     end    
--     table.insert(data[fr],da) 
--   end
--   
--   local tL,minKey, maxKey = tabLen(data)
--   local F = maxKey
--   local N = 255
--   
--   local labs = torch.IntTensor(N, F):fill(0)
--   
--   print(data)
-- --   abort()
--   -- Now create a FxN tensor with labels
--   for t in pairs(data) do
--     local cnt=0
--     for i = 1, N do
--       cnt=cnt+1
--       -- 	local gtID, gtBBox = orderedKeys[i], gt[t+ts-1][ orderedKeys[i] ]
--       local gtBBox = data[t][i]
--       if gtBox then
-- 	labs[cnt][t] = gtBox[1][1]
--       end
--     end
--   end
--   print(labs)
--   
--   -- remove zero rows
-- --   labs = cleanDataTensor(labs)
--   local N,F = getDataSize(labs)
--   local exTar = torch.ne(torch.sum(labs,2),0):reshape(N) -- binary mask
--   local nTar = torch.sum(exTar) -- how many true tracks  
--   
--   local newData = torch.Tensor(nTar,F)
--   local cnt = 0
--   for i=1,N do
--     if exTar[i] > 0 then
--       cnt=cnt+1
--       newData[{{cnt},{}}]=labs[{{i},{}}]
--     end
--   end
--   
-- 
--   
--   return newData
-- end	
	

--------------------------------------------------------------------------
--- Get results of a sequence as a tensor. NOTE All tracks are retrieved.
-- @param seqName 	The name of the sequence.
-- @param resDir	Folder where results are stored.
function getResTracks(seqName, resDir)
--   ts = ts or 1 	-- default starting frame
  local nDim = 4		-- number of dimensions
  
  local res = getRes(seqName, resDir)	-- get raw GT table
  
  local imgHeight, imgWidth = getImgRes(seqName) -- required for normalization
  imgWidth = 1
  local tL,minKey, maxKey = tabLen(res)
  
  if tL == 0 then return torch.zeros(0,0,nDim) end
  local F = maxKey
  local N = 255
  
  local tracks = torch.zeros(N, F, nDim)

  -- Now create a FxN tensor with states
  for t=1,F do
    local cnt=0
    for i = 1, N do
      cnt=cnt+1	
      if res[t] ~= nil then
	local gtBBox = res[t][i]
	if gtBBox then 
	  for dim = 1,nDim do
	    tracks[cnt][t][dim]=gtBBox[1][dim] / imgWidth 	-- normalize to [0,1]
	  end
	end
      end
    end
  end

  tracks = cleanDataTensor(tracks) -- remove all-zero tracks
  return tracks
end

--------------------------------------------------------------------------
--- Crop GT or detection tensors to a specific time window.
-- @param data	The data tensor.
-- @param ts	First frame.
-- @param te	Last frame.
function selectFrames(data, ts, te, clean)
--   if clean == nil then clean = true end
  local N,F,D = getDataSize(data)		-- data size
  assert(ts<=te,"Start point must be <= end point")
  if ts<1 or te>F then print("Warning! Frame selection outside bounds") end

  ts=math.max(1,ts);  te=math.min(F,te) -- make sure we are within bounds
  data=data:narrow(2,ts,te-ts+1) 	-- perform selection

  
  if clean == nil or clean == true then
--     data = cleanDataTensor(data)		-- clean zero tracks TODO
  end
  return data    
end

--------------------------------------------------------------------------
--- Remove all-zero tracks (rows).
-- @param data	The data Tensor N x F x Dim.
function cleanDataTensor(data)
  local N,F,D = getDataSize(data)
  local exTar = torch.ne(torch.sum(data:narrow(3,1,1),2),0):reshape(N) -- binary mask
  local nTar = torch.sum(exTar) -- how many true tracks  
  
  local newData = torch.Tensor(nTar,F,D)
  local cnt = 0
  for i=1,N do
    if exTar[i] > 0 then
      cnt=cnt+1
      newData[{{cnt},{}}]=data[{{i},{}}]
    end
  end
  
  return newData
end

--------------------------------------------------------------------------
--- Get detections of a specific sequence as a table.
-- @param seqName	The name of the sequence.
-- @return 		A table containing the detection bounding boxes
function getDetections(seqName, detfile)
  detfile = detfile or getDataDir() .. seqName .. "/det/det.txt"
  if lfs.attributes(detfile) then 
--     print("Detections file ".. detfile)
  else
    error("Error: Detections file ".. detfile .." does not exist")
  end     
  return readTXT(detfile, 2) -- param #2 = detections
end

--------------------------------------------------------------------------
--- Get detections of a specific sequence as a tensor.
-- @param seqName	The name of the sequence.
-- @return 		A Nd x F x 4 Tensor
-- @see getDetections
-- @see getDetectionsVec
function getDetTensor(seqName, detfile)
  local detTab = getDetections(seqName, detfile)
  local imgHeight, imgWidth = getImgRes(seqName)
  imgWidth = 1
  local F = tabLen(detTab)
  local nDim = 4 	-- bounding box
  
  F=0;
  local maxDetPerFrame = 0
  for k,v in pairs(detTab) do 
    if k>F then F=k end 
    if tabLen(v)>maxDetPerFrame then maxDetPerFrame = tabLen(v) end
  end
  
  local detections = torch.zeros(maxDetPerFrame,F, nDim)
  
  for t=1,F do
    if detTab[t] then -- skip if no detections present in frame 
      for detID,detBBox in pairs(detTab[t]) do      
-- 	detections[detID][t] = getFoot(detBBox)[1][1] / imgWidth  
	detections[{{detID},{t},{}}] = detBBox:narrow(2,1,nDim) / imgWidth  	
      end
    end
  end
  local Ndet,Fdet,Ddet = getDataSize(detections)
  if Ndet<1 then detections=torch.zeros(1,F,nDim); maxDetPerFrame=1 end
  
  
  
  -- pad detections with empty frames at end if necessary
  local Ngt,Fgt,Dgt = getDataSize(getGTTracks(seqName))
  if Fgt>F then
    detections=detections:cat(torch.zeros(maxDetPerFrame,Fgt-F,nDim),2)
  end
--   print(detections)
  
  return detections
end


--------------------------------------------------------------------------
--- Selects a specific state dimension from a NxFxD tensor.
-- @return Only one dimension, i.e. an NxF tensor
function selectStateDim(data, dim)
  return data:narrow(3,dim,1):reshape(data:size(1),data:size(2))
end



--------------------------------------------------------------------------
--- Intersection over union of two boxes.
-- The boxes are given as 1D-Tensors of size 4 with xc,yc,w,h
function boxIoU(box1, box2)
  local ax1=box1[1]-box1[3]/2; local ax2=ax1+box1[3]; 
  local ay1=box1[2]-box1[4]/2; local ay2=ay1+box1[4];
  
  local bx1=box2[1]-box2[3]/2; local bx2=bx1+box2[3]; 
  local by1=box2[2]-box2[4]/2; local by2=by1+box2[4];
  
--   print(ax1, ax2, ay1,ay2)
--   print(bx1, bx2, by1,by2)
--   if ax1>ax2 or ay1>ay2 then print(string.format('WARNING: wrong bbox: %f, %f, %f, %f', ax1,ax2,ay1,ay2)) end
--   if bx1>bx2 or by1>by2 then print(string.format('WARNING: wrong bbox: %f, %f, %f, %f', bx1,bx2,by1,by2)) end
  -- TODO IMPORTANT! Boxes can be wrong sometimes!
  
  local hor = math.min(ax2, bx2) - math.max(ax1,bx1) -- horiz. intersection
  
  local ver = 0
  if hor>0 then
    ver = math.min(ay2, by2) - math.max(ay1,by1) -- vert. intersection
  end
  local isect = hor*ver
  local union = box1[3]*box1[4] + box2[3]*box2[4] - isect
	
  if union <=0 then return 0 end

  return isect / union
  
end

--------------------------------------------------------------------------
--- Find closest track to one particular detection. Depending on the 
-- state dimension, the distance is either absolute distance (1D),
-- Euclidean distance (2D), or 1-IoU (4D)
-- @param det		Detection (A 1D, 2D, or 4D vector)
-- @param tracks	All tracks in one frame (Nx1xD)
-- @return 		Closest distance
-- @return 		Closest tack ID
-- @return		All distances
function findClosestTrack(det, tracks)
  -- Find out state dimension
  local N, F, stateDim = getDataSize(tracks)
  assert(stateDim == det:size(1),'state dimensions do not agree')

  
  local sDist={};  local cTr = {}; local distVec = torch.zeros(N)
  if stateDim == 1 then		-- Absolute distance
    tracks = tracks:reshape(N) -- make an N-D vector
    -- clone provided detection N times to match tracks vector
--     print(det)
    local cloneDet = torch.repeatTensor(det,N,1)

    distVec = torch.abs(cloneDet-tracks)
    sDist,cTr = torch.min(distVec,1) -- closest track (dist, ID)
    sDist=sDist:squeeze(); cTr = cTr:squeeze() -- squeeze for dx1-size to d-size     

  elseif stateDim >= 2 then
    tracks = tracks:reshape(N, stateDim):narrow(2,1,2)
    local cloneDet = torch.repeatTensor(det, N, 1):narrow(2,1,2)

    distVec = torch.abs(cloneDet-tracks)
    distVec = torch.pow(distVec, 2)
    distVec = torch.sum(distVec,2)
    distVec = torch.sqrt(distVec)
    sDist,cTr = torch.min(distVec,1) -- closest track (dist, ID)
    sDist=sDist:squeeze(); cTr = cTr:squeeze() -- squeeze for dx1-size to d-size     
    
    
--[[  elseif stateDim == 4 then
    tracks = tracks:narrow(3,1,2):reshape(N, 2)
    local cloneDet = torch.repeatTensor(det:narrow(1,1,2), N, 1)
    
    distVec = torch.abs(cloneDet-tracks)
    distVec = torch.pow(distVec, 2)
    distVec = torch.sum(distVec,2)
    distVec = torch.sqrt(distVec)
    sDist,cTr = torch.min(distVec,1) -- closest track (dist, ID)
    sDist=sDist:squeeze(); cTr = cTr:squeeze() -- squeeze for dx1-size to d-size      ]]   
    

  else
    error('distance computation for state dim '..stateDim..' not implemented')
  end
  
  return sDist, cTr, distVec
end

--------------------------------------------------------------------------
--- Find closest track to one particular detection (4D states only). 
-- The distance is 1-IoU (4D)
-- @param det		Detection (4D vector)
-- @param tracks	All tracks in one frame (Nx1x4)
-- @return 		Closest distance
-- @return 		Closest tack ID
-- @return		All distances
function findClosestTrackIOU(det, tracks)
    -- Find out state dimension
  local N, F, stateDim = getDataSize(tracks)
  assert(stateDim == det:size(1),'state dimensions do not agree')

  
  local sDist={};  local cTr = {}; local distVec = torch.zeros(N)
  if stateDim == 4 then
    sDist = 1 -- largest distance
    for i=1,N do
      if det[1] ~= 0 and tracks[i]:squeeze()[1] ~= 0 then -- ignore dummies      
	ioud = 1-boxIoU(det,tracks[i]:squeeze())
	distVec[i] = ioud
-- 	print(i,ioud)
	if ioud < sDist then
	  sDist = ioud
	  cTr = i
	end
      end
    end

  else
    error('distance computation for state dim '..stateDim..' not implemented')
  end
  
  return sDist, cTr, distVec
end






--------------------------------------------------------------------------
--- Returns number of rows, number of columns (optional number of dim3) of a tensor
function getDataSize(data)
  -- rows, columns, dimensions
  local R, C, D = 0, 0, 0
  
  if data:nDimension()>=1 then R=data:size(1) end
  if data:nDimension()>=2 then C=data:size(2) end  
  if data:nDimension()>=3 then D=data:size(3) end
  return R,C,D
end

--------------------------------------------------------------------------
--- Get the bottom center point of a box
-- @param bbox	A 1x4 tensor describing the bbox as [bx,by,bw,bh]
-- @return A 1x2 tensor giving the x,y position of the feet
function getFoot(bbox) 
  local pos = torch.Tensor(1,2):fill(0)
  pos[1][1] = bbox[1][1] + bbox[1][3]/2
  pos[1][2] = bbox[1][2] + bbox[1][4]
  return pos
end

--------------------------------------------------------------------------
--- Get the center point of a box
-- @param bbox	A 1x4 tensor describing the bbox as [bx,by,bw,bh]
-- @return A 1x2 tensor giving the x,y position of the center
function getCenter(bbox) 
  local pos = torch.Tensor(1,2):fill(0)
  pos[1][1] = bbox[1][1] + bbox[1][3]/2
  pos[1][2] = bbox[1][2] + bbox[1][4]/2
  return pos
end

--------------------------------------------------------------------------
-- Returns image height and width for a given sequence.
-- @param seqName 	Name of sequence
-- @return height
-- @return width
-- @return nChannels
function getImgRes(seqName)
  if seqName == 'Synth' then return 1000,1000,3 end
  
  local imFile = getDataDir() .. seqName .. "/img1/000001.jpg"
  local im = image.load(imFile)
  h,w,c =im:size(2), im:size(3), im:size(1) 
  im = nil
  collectgarbage() -- manual garbage collection to prevent mem. leak
  return h,w,c  -- height x width x nChannel
end

--------------------------------------------------------------------------
--- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
-- @author Andrej Karpathy
-- TODO Modify to accept and return parameter
function checkCuda()  
  if opt.gpuid >= 0 and opt.opencl == 0 then
      local ok, cunn = pcall(require, 'cunn')
      local ok2, cutorch = pcall(require, 'cutorch')
      if not ok then print('package cunn not found!') end
      if not ok2 then print('package cutorch not found!') end
      if ok and ok2 then
	  print('using CUDA on GPU ' .. opt.gpuid .. '...')
	  cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
	  cutorch.manualSeed(opt.seed)
      else
	  print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
	  print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
	  print('Falling back on CPU mode')
	  opt.gpuid = -1 -- overwrite user setting
      end
  end  

	if opt.gpuid >= 0 and opt.opencl == 1 then
		local ok, cunn = pcall(require, 'clnn')
		local ok2, cutorch = pcall(require, 'cltorch')
		if not ok then print('package clnn not found!') end
		if not ok2 then print('package cltorch not found!') end
		if ok and ok2 then
			print('using OpenCL on GPU ' .. opt.gpuid .. '...')
			cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
			torch.manualSeed(opt.seed)
		else
			print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
			print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
			print('Falling back on CPU mode')
			opt.gpuid = -1 -- overwrite user setting
		end
	end

end

--------------------------------------------------------------------------
--- Experimental: moves a torch tensor to gpu if needed. Otherwise, converts it to float
function dataToGPU(data)
--   if true then return data end
  
--   local timerF = torch.Timer()
  if data==nil then return nil end
  data=data:float()
  if opt.gpuid >= 0 and opt.opencl == 0 then
    data = data:float():cuda()
  end
  
  if opt.gpuid >= 0 and opt.opencl == 1 then
    data = data:cl()
  end
--   if opt.profiler>0 then profUpdate('dataToGPU', timerF:time().real) end
  return data
end

--------------------------------------------------------------------------
--- Returns number of elements in a table, as well as min. and max. key
-- @return Length of table (number of keys)
-- @return min key
-- @return max key
function tabLen(tab)
  local count = 0
  local minKey = 1e5
  local maxKey = -1e5
  for key in pairs(tab) do 
    count = count + 1 
    if key < minKey then minKey = key end
    if key > maxKey then maxKey = key end
  end
  return count, minKey, maxKey
end

--------------------------------------------------------------------------
--- Pads a tensor with zeros along a certain dimension if necessary
-- @param data	The tensor.
-- @param N	The desired dimensionality
-- @param dim 	The dimension along which to pad
function padTensor(data, N, dim)
  assert(data:nDimension()<=3, "At most 3-dim tensor expected")
  local R,C,D = getDataSize(data)
  
  if N<=data:size(dim) then return data end 	-- nothing to do
  
  if dim == 1 then
    return data:cat(torch.zeros(N-R,C,D),1)
  elseif dim == 2 then
    return data:cat(torch.zeros(R,N-C,D),2)
  elseif dim == 3 then
    return data:cat(torch.zeros(R,C,N-D),3)
  end
  
end


--------------------------------------------------------------------------
--- Hungarian Algorithm
function hungarianL(mat)
--   print(mat)
  assert(mat:nDimension() == 2, 'cost matrix must be 2D')
--   assert(mat:size(1) == mat:size(2), 'cost matrix must be square')
  
  local matTab = mat:clone():totable()
--   local timer = torch.Timer()
  local ass = hungarian(matTab)
  local retTen = torch.ones(tabLen(ass),2)

  local cnt=0
  for k,v in pairs(ass) do
    cnt=cnt+1
    retTen[cnt][1] = k
    retTen[cnt][2] = v
  end

  return retTen

end      

--------------------------------------------------------------------------
--- Returns mean width of all detections
function getMeanDetWidth(detections)
  assert(detections:size(3)>=3, 'need 3-dim detections at least')
  
  local detEx = detections[{{},{},{1}}]:ne(0)
  local detWidths = detections:sub(1,-1,1,-1,3,3)
  local Ndet,Fdet,Ddet = getDataSize(detections)
  detWidths = detWidths:reshape(Ndet,Fdet)    
  local meanWidth = torch.mean(detWidths[detEx])
  return meanWidth
end

--------------------------------------------------------------------------
--- Returns pid of X Server if available, nil otherwise
function xAvailable()
  if lfs.attributes('/home/h3/','mode') then return false end	-- network
  
  local f = io.popen('pidof X')
  local t = f:read()
  f:close()
  
  local s = true
  if opt.suppress_x ~= nil and opt.suppress_x > 0 then s = false end
  
  return t and s
end


----------------------------------------
-- Learn generative trajectory  model --
function learnTrajModel(trTable)
  local tL = tabLen(trTable)
  _,_,D = getDataSize(trTable[1])
  
  -- number of features for each trajectory
  -- 2 start / end frame
  -- 2 * D start / end state
  -- D mean velocity
  -- 
  
  local nFeat = 2 + 2*D + D
--   local nFeat = 2 + 2*D + 0
  
--   print(m)
--   abort()
  
  local trackModels = {}
  
  for s=1,tL do
    local m = torch.zeros(1,nFeat) -- dummy row, rest is catted on bottom
    local tracks = trTable[s]
    
    N,F,D = getDataSize(tracks)    
--     print(s, N,F,D)
--     print(tracks)
    
    for i=1,N do      
      
      local tr= tracks[i]:narrow(2,1,1) -- 2 because [i] removes first dim      
      local exfr = torch.find(tr:squeeze()) -- 1D vec with frame nums
      local trLength = exfr:nElement()
      if trLength > 0 then
	
	local featTab = {}
	
	-- start, end point
	local s = exfr[1]
	local e = exfr[-1]
	
	table.insert(featTab, s)
	table.insert(featTab, e)
	
	-- locations
	for d=1,D do
  -- 	print(i,s,d)
	  local sd = tracks[i][s][d]
	  local ed = tracks[i][e][d]
	  table.insert(featTab, sd)
	  table.insert(featTab, ed)	  	  
	end	
	
	-- mean velocities
	for d=1,D do
	  local mv = (tracks[i][e][d] - tracks[i][s][d]) / (e-s+1)
-- 	  print(mv)
	  table.insert(featTab, mv)
	end
	
-- 	print(featTab)
	local featVec = torch.Tensor(featTab):resize(1,nFeat)
-- 	print(m)
-- 	print(featVec)
	m = m:cat(featVec,1)
      
      end
    end
--     print(tracks)
--     print(m)
    
    -- dont bother if only dummy sample
    if m:size(1)>1 then
      m = m:sub(2,-1) -- remove first dummy row
--       print(m)
      mmean = torch.mean(m,1)
      mstd = torch.std(m,1)
      if N<=1 then mstd = torch.mul(mmean, 1)/2 end -- avoid NaNs if one track only
      
      -- HACK EXPERIMENTAL
      -- transfer vars from y to x
      if D>1 then
-- 	for d=1,stateDim-1 do
-- 	  local toRep = 4+(2*d)-1
-- 	  mstd[{{},{toRep}}]=mstd[{{},{3}}]/2 -- start
-- 	  local toRep = 4+(2*d)
-- 	  mstd[{{},{toRep}}]=mstd[{{},{4}}]/2 -- end
-- 	end
      end
      table.insert(trackModels, torch.cat(mmean, mstd, 1))
      
    end
  end
  
--  for k,v in pairs(trackModels) do print(k) print(v) end  
--  abort()
  
  return trackModels
--   
end

--------------------------------------------------------------------------
--- Return a sample from a 1D normal distribution
function sampleNormal(mean, std)
--   print(torch.randn(1):squeeze())
  return std*torch.randn(1):squeeze() + mean
end



--------------------------------------------------------------------------
--- Updates a global profiling table
function profUpdate(fname, ftime)
  if profTable[fname] == nil then
    profTable[fname] = torch.Tensor({0,0})
  end
  profTable[fname] = profTable[fname] + torch.Tensor({ftime, 1})
end




--------------------------------------------------------------------------
--- Runs model on all training sequences of the 2DMOT15 Challenge
function runMOT15(runScript, modelName, modelSign, seqs)
  -- matlab
  local cmdstr=''
  if seqs==nil then
    cmdstr = string.format('matlab -nodesktop -nosplash -nojvm -r "cd matlab; runRNN(\'%s\',\'%s\',\'%s\',\'%s\'); quit;"',
      modelName,modelSign,opt.eval_conf,runScript)
  else    
    cmdstr = string.format('matlab -nodesktop -nosplash -nojvm -r "cd matlab; runRNN(\'%s\',\'%s\',\'%s\',\'%s\',%s); quit;"',
      modelName,modelSign,opt.eval_conf,runScript,cellArray(seqs))
  end

  local file = assert(io.popen(cmdstr, 'r'))

  local mets = torch.ones(1,14) * (-100)
  while true do
    local line = file:read()				-- read line    
    if line == nil then break end				-- final line, break  
    print(line)						-- print for logging
    local matchfunc = string.gmatch(line, "([^,]+)") -- split string on commas
    local fields = {}		-- make table from split string
    for str in matchfunc do table.insert(fields, str) end  
  
    -- if correct line (N1,N2,...N14), make tensor
    if tabLen(fields) == 14 then mets = torch.Tensor(fields):view(1,-1) end  
  end
  
  return mets
end

--------------------------------------------------------------------------
--- Keep a file with progress wrt. mota
-- @param XX 	XX TODO
-- @return 	True if new all-time best MOTA was found, false otherwise.
function writeBestMOTA(newMOTA, modelName, modelSign, filename) -- update all time best
  
  local prevBest = -100		-- dummy value
  filename = filename or 'bestmota.txt'
  ret = false
  
  if lfs.attributes(filename,'mode') then 			-- virtual machine
    pm('Best mota file exists...',2)

    local file = assert(io.open(filename, 'r'))

    local fields = {}
    while true do
      local line = file:read()				-- read line      
      if line == nil then break end				-- final line, break  
--       print(line)						-- print for logging
      local matchfunc = string.gmatch(line, "([^%s]+)") -- split string on whites
      fields = {}		-- make table from split string
      for str in matchfunc do table.insert(fields, str) end      
    end    
    prevBest = tonumber(fields[1])			-- update prevBest (last entry)
    

  end
  pm('Previous best MOTA is '..prevBest,2)
  
  
  if newMOTA > prevBest then
    pm('Writing new MOTA '..newMOTA,2)
    local dateStr = os.date('%Y-%m-%d_%H:%M')  -- e.g. 2015-09-22_08:43
    
    local file = assert(io.open(filename,'a+'))
    file:write(string.format('%5.1f %20s %10s %15s\n',newMOTA,dateStr,modelName,modelSign))
    file:close()
    
    ret = true
  end
  return ret
  
end

--------------------------------------------------------------------------
--- Get options from a config file
function parseConfig(configFile, opt)
  local file = assert(io.open(configFile, 'r'))
  
  -- go through all lines
  while true do
    local line = file:read()				-- read line      
    if line == nil then break end				-- final line, break  
    local matchfunc = string.gmatch(line, "([^%s]+)") -- split string on white spaces
    
    local key = nil
    local val
    for str in matchfunc do
      if key==nil then key = str		-- new line, set key
      else					-- handle value
	local numparam = tonumber(str)		-- if it is a number
	if numparam ~= nil then 		
	  val = numparam			-- set as number
	else val = str end			-- else set as string
      end
--       print(key,val)
--       sleep()
      -- update key and value if both are not nil
      if key~=nil and val~=nil then opt[key] = val end
    end
  end  
--   abort()
  return opt
end

--------------------------------------------------------------------------
--- Breaks a full path into [path, filename, extension]
-- @return 		Full path to file
-- @return 		Filename (without extension)
-- @return 		File extension
function fileparts(str)  
  local str_len = string.len(str)	-- length of string
  
  -- slash/backslashes  
  local last_slash = (string.find(string.reverse(str),'[\\/]'))
  if last_slash == nil then last_slash = 0
  else last_slash = str_len - last_slash + 1 end
  
  -- dot
  local last_dot = ((string.find(string.reverse(str),'%.')))
  if last_dot == nil then last_dot = str_len + 1 
  else last_dot = str_len - last_dot + 1 end
  
  
  
  local fp, fn, fe = '', '', ''
  fp = string.sub(str, 1, last_slash)
  fn = string.sub(str, last_slash+1, last_dot-1)
  fe = string.sub(str, last_dot+1, str_len)
  
  return fp,fn,fe
end


-- from https://stackoverflow.com/questions/15706270/sort-a-table-in-lua
function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

-- http://lua-users.org/wiki/CopyTable
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function getVelocities(tracks)
  velTab = {}
  
  -- tracks is a table of NxFxD tensors
  for k,v in pairs(tracks) do
    local N,F,D = getDataSize(v)
    local vel = torch.zeros(N,F,D)
    
    -- main part, bi-dir finite diff.
    for t=2,F-1 do
      for i=1,N do
	
	local prevL = v[i][t-1]
	local thisL = v[i][t]
	local nextL = v[i][t+1]
	local exPrev = torch.sum(torch.abs(prevL))~=0
	local exThis = torch.sum(torch.abs(thisL))~=0
	local exNext = torch.sum(torch.abs(nextL))~=0
-- 	print(i,t, prevL, nextL)
	if exPrev and exNext then
	  vel[i][t] = (nextL-prevL)*0.5
	elseif exThis and exNext then
	  vel[i][t] = (nextL-thisL)
	elseif exPrev and exThis then
	  vel[i][t] = (thisL-prevL)
	end
-- 	print(nextL-prevL)
-- 	print(vel)
      end
    end
    
    for i=1,N do
      -- first frame
      local t=1
      local thisL = v[i][t]
      local nextL = v[i][t+1]
      if torch.sum(torch.abs(thisL))~=0 and torch.sum(torch.abs(nextL))~=0 then
	vel[i][t] = nextL - thisL
      end
      
      -- last frame
      local t=F
      local thisL = v[i][t]
      local prevL = v[i][t-1]
      if torch.sum(torch.abs(thisL))~=0 and torch.sum(torch.abs(prevL))~=0 then
	vel[i][t] = thisL - prevL
      end
      
      
    end
    
--     print(vel)
    velTab[k] = vel
  end
  
  return velTab 
end



function showLocals()
-- function locals()
--   local variables = {}
--   local idx = 1
--   while true do
--     local ln, lv = debug.getlocal(2, idx)
--     if ln ~= nil then
--       variables[ln] = lv
--     else
--       break
--     end
--     idx = 1 + idx
--   end
--   return variables
-- end
-- 
-- 
-- print(locals())
-- abort()
end

--------------------------------------------------------------------------
--- Move all detection 'up' to fill ID slots 1,2,3...
-- This may also reshuffle detections
function moveUp(detections)
  local Ndet,Fdet,Ddet=getDataSize(detections)
  local detOrig = detections:clone()
  detections:fill(0)
  
  for t=1,Fdet do
    local cnt=0
    for i=1,Ndet do
      if detOrig[i][t][1] ~= 0 then
	cnt=cnt+1
	detections[cnt][t] = detOrig[i][t]
      end
    end
  end  
  return detections
end

-- TODO docs
function getLabelsFromLL(data, HA)
  local loctimer=torch.Timer()
  local N,F,L = getDataSize(data)
  local lab = torch.IntTensor(N,F):fill(0)
  
  if HA==nil then HA = false end
--   print(data)
--   print(lab)
--   local HA = true
--   local HA = false
--   print(HA)
--   abort('HA')
  if HA then
    -- hungarian 1-to-1
--     assert(N==L, 'nTargets must be equal nLabels')
    for t=1,F do
      
--       print(data:narrow(2,t,1))
--       print(data:narrow(2,t,1):reshape(opt.max_n, opt.max_m))
--       print(data)
      local distmat = data:narrow(2,t,1):reshape(maxTargets, nClasses)
--       print(distmat)
--       abort()
      ass = hungarianL(-distmat)
--       print(ass)
--       print(ass[{{},{2}}]:t())
      
      lab[{{},{t}}] = ass[{{},{2}}]:t()
--       for i=1,N do
-- 	lab[i][t] = ass[i][2]
-- 	lab[ass[i][2]][t] = i
--       end
    end
--     abort()
  else
    -- max
    for i=1,N do
      for t=1,F do
  --       print(i,t)
	local thisPred = data[{{i},{t},{}}]:reshape(1,L)
  --       print(thisPred)
	mv,mi = torch.max(thisPred,2)
  --       print(mi)
	lab[i][t] = mi:squeeze()
      end
    end
  end
  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end 
--   print(lab)
  return lab
end


function reshuffleDetsAndLabels(detections, labels, t)

  local detclone, labclone = detections:clone(), labels:clone()
--   print(detections:squeeze())
--   print(labels)


  local miniBatchSize = detclone:size(1)/opt.max_m
  if t~= nil then
--     if miniBatchSize>1 then error('IMPLEMENT MULTIBATCH') end
--     detInFrame = detections:narrow(2,t+1,1)
    
    for mb=1,miniBatchSize do
      local mbOffset=(mb-1)*opt.max_n
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb

      local mbOffsetD=(mb-1)*opt.max_m
      local mbStartD = opt.max_m * (mb-1)+1
      local mbEndD =   opt.max_m * mb
      
      local detInBatch = detections[{{mbStartD,mbEndD},{t+1},{}}]
      local labInBatch = labels[{{mbStart,mbEnd},{t+1}}]
    
      local detInBatch, scrindex = reshuffleDets(detInBatch)
--       print(scrindex)
--       abort('a')
      
      detections[{{mbStartD,mbEndD},{t+1},{}}] = detInBatch
      local N,F = getDataSize(labInBatch)
      lorig = labInBatch:clone()
      for i=1,N do
-- 	print(i, mbOffset, scrindex[i][1])
-- 	print(labels)
-- 	print(lorig)
-- 	print(i,t)
	labels[i+mbOffset][t+1] = lorig[scrindex[i][1]]
-- 	labels[scrindex[i][1]+mbOffset][t+1] = lorig[i]
      end
    end
  else
    -- leave first frame untouched
-- -- --     local locdets, loclabs = detections:clone(), labels:clone()
-- -- --     detections, scrindex = reshuffleDets(detections)        
-- -- --     lorig = labels:clone()
-- -- --     local N,F = getDataSize(labels)
-- -- -- --     print(scrindex)
-- -- -- --     abort()
-- -- --     for i=1,N do 
-- -- --       for t=2,F do
-- -- -- 	labels[i][t] = lorig[scrindex[i][t]][t]
-- -- --       end 
-- -- --     end
-- -- --     detections[{{},{1},{}}] = locdets:narrow(2,1,1)
-- -- --     labels[{{},{1}}] = loclabs:narrow(2,1,1)
--     detections, labels = locdets, loclabs
    
    local locdets, loclabs = detections, labels
    
    for mb=1,miniBatchSize do
      local mbOffset=(mb-1)*opt.max_n
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb

      local mbOffsetD=(mb-1)*opt.max_m
      local mbStartD = opt.max_m * (mb-1)+1
      local mbEndD =   opt.max_m * mb
      
      local batchDets = locdets[{{mbStartD,mbEndD},{},{}}]
      local batchLab = loclabs[{{mbStart,mbEnd},{}}]

--       printDim(batchDets)
--       print(batchLab)
      
--       print(mb)
      batchDets, scrindex = reshuffleDets(batchDets)     
--       printDim(batchDets)
--       print(scrindex)
      
      
      -- WARNING THIS SORTS DIFFERENTLY
      local N,F = getDataSize(scrindex)
      local newScrindex = torch.IntTensor(maxTargets,F)
      for t=1,F do 
-- 	print(scrindex[{{},{t}}])
	local _,srted = torch.sort(scrindex[{{},{t}}]:reshape(maxDets))
	srted = srted:reshape(maxDets)
	newScrindex[{{},{t}}] = srted[{{1,maxTargets}}]
	
-- 	for i=1,maxTargets do  	
-- 	print(srted)
-- 	abort()
--   	scrindex[{{},{t}}] = srted
-- 	newScrindex[scrindex[i][t]][t] = i
-- 	  newScrindex[i][t] = scrindex[i][t]
-- 	end 
      end
      scrindex = newScrindex:clone()
--       print(scrindex)      
--       abort('r')
--       print(locdets:squeeze())
--       print(loclabs)
	
      locdets[{{mbStartD,mbEndD},{},{}}]=batchDets:clone()
      loclabs[{{mbStart,mbEnd},{}}] = scrindex:clone()
--       print(locdets:squeeze())
            
--       lorig = batchLab:clone()
--       local N,F = getDataSize(batchLab)
--       for i=1,N do 
-- 	for t=2,F do
-- 	  batchLab[i][t] = lorig[scrindex[i][t]][t]
-- 	end 
--       end            
--       print(batchLab)
--       print(loclabs)
--       print('---')
    end
    
    -- leave first frame untouched
    detections[{{},{1},{}}] = detclone:narrow(2,1,1)
    labels[{{},{1}}] = labclone:narrow(2,1,1)
    
    
  end
    
--   print(detections:squeeze())
--   print(labels)
--   abort()
  return detections, labels
end


function reshuffleDetsAndLabels2(detections, labels, t)

  local detclone, labclone = detections:clone(), labels:clone()
--   print(detections:squeeze())
--   print(labels)


  local miniBatchSize = detclone:size(1)/opt.max_m
  if t~= nil then
--     if miniBatchSize>1 then error('IMPLEMENT MULTIBATCH') end
--     detInFrame = detections:narrow(2,t+1,1)
    
    for mb=1,miniBatchSize do
      local mbOffset=(mb-1)*opt.max_m
      local mbStart = opt.max_m * (mb-1)+1
      local mbEnd =   opt.max_m * mb
      
      local detInBatch = detections[{{mbStart,mbEnd},{t+1},{}}]
      local labInBatch = labels[{{mbStart,mbEnd},{t+1}}]
    
      local detInBatch, scrindex = reshuffleDets(detInBatch)
--       print(scrindex)
--       abort('a')
      
      detections[{{mbStart,mbEnd},{t+1},{}}] = detInBatch
      local N,F = getDataSize(labInBatch)
      lorig = labInBatch:clone()
      for i=1,N do
	labels[i+mbOffset][t+1] = lorig[scrindex[i][1]]
      end
    end
  else

    local locdets, loclabs = detections, labels
    
    for mb=1,miniBatchSize do
      local mbOffset=(mb-1)*opt.max_m
      local mbStart = opt.max_m * (mb-1)+1
      local mbEnd =   opt.max_m * mb
      
      local batchDets = locdets[{{mbStart,mbEnd},{},{}}]
      local batchLab = loclabs[{{mbStart,mbEnd},{}}]

--       print(mb)
      batchDets, scrindex = reshuffleDets(batchDets)     

      locdets[{{mbStart,mbEnd},{},{}}]=batchDets:clone()
      loclabs[{{mbStart,mbEnd},{}}] = scrindex:clone()

    end
    
    -- leave first frame untouched
    detections[{{},{1},{}}] = detclone:narrow(2,1,1)
    labels[{{},{1}}] = labclone:narrow(2,1,1)
    
    
  end

  return detections, labels
end

function orderTracks(tracks, lab)
  local N,F,D = getDataSize(tracks)
  local NL, FL = getDataSize(lab)
  assert(N==NL and F==FL, 'tracks and labs must be equal size')
  
  
  newTracks = tracks:clone():fill(0)
  for t=1,F do
    for i=1,N do
      newTracks[lab[i][t]][t] = tracks[i][t]
    end
  end
  return newTracks
  
end



function zeroTensor1(a)
  if opt.gpuid >= 0 and opt.opencl == 0 then
    return torch.CudaTensor(a)
  elseif opt.gpuid < 0 and opt.opencl == 1 then
    return torch.zeros(a):cl()
  else
    return torch.zeros(a):float()
  end 
end

function zeroTensor2(a,b)
  if opt.gpuid >= 0 and opt.opencl == 0 then
    return torch.CudaTensor(a,b)
  elseif opt.gpuid < 0 and opt.opencl == 1 then
    return torch.zeros(a,b):cl()
  else
    return torch.zeros(a,b):float()
  end 
end

function zeroTensor3(a,b,c)
  if opt.gpuid >= 0 and opt.opencl == 0 then
    return torch.CudaTensor(a,b,c)
  elseif opt.gpuid < 0 and opt.opencl == 1 then
    return torch.zeros(a,b,c):cl()
  else
    return torch.zeros(a,b,c):float()
  end
  
end





function getImSizes(seqTable, imSizes)
  if imSizes==nil then imSizes = {} end
  for k, v in pairs(seqTable) do
    if imSizes[v] == nil then imSizes[v] = {}
      if v=='art' then imSizes[v]['imH'], imSizes[v]['imW'] = 1,1
      else
	imSizes[v]['imH'], imSizes[v]['imW']= getImgRes(v)    
      end
    end
  end
  local v = 'Synth' imSizes[v] = {}
  imSizes[v]['imH'], imSizes[v]['imW']= getImgRes(v)    
  return imSizes
end





function cellArray(tab)
  local str = '{'
  for k,v in pairs(tab) do
    str = str..'\''..v..'\''    
  end
  str = str..'}'
  str = string.gsub(str, "''", "','") -- add commas
  return str
end

function squeezeLabels(tab, singleBatch)
  -- remove labels from non-existing targets
  for k,v in pairs(tab) do
    local newLab = torch.IntTensor(1, opt.temp_win):fill(0)
    nB = opt.mini_batch_size
    if singleBatch then nB = 1 end
    for mb=1,nB do
      local mbStart = opt.max_m * (mb-1)+1
      newLab = newLab:cat(v[{{mbStart,mbStart+opt.max_n-1},{}}],1)
    end
    newLab=newLab:sub(2,-1)
    tab[k]=newLab
  end
  return tab
end


function getPredEx(ex)
  if ex==nil then return nil end
  local thr = opt.ex_thr or 0.5
  
  
  local N,T = getDataSize(ex) -- T is missing frames
  local exTensor = {} 
  if ex ~= nil then
    local nEx = 0
    if type(ex)=='table' then 
      error('hmm')
--       nEx = tabLen(ex)    
--       exTensor = zeros(N,nEx) -- NxF tensor with 1,2 existance classes
--       for t,_ in pairs(ex) do
-- 	mv,mi = torch.max(ex[t],2)
-- 	if mi:squeeze()==1 then exTensor[{{},{t}}] = 1 end
--       end
    else -- ex is an NxFx2 tensor      
-- -- --       exTensor = torch.zeros(N,T) -- NxF tensor with 1,2 existance classes
-- -- --       local allTwos = torch.ones(1,T):long() * 2
-- -- --       for i=1,N do
-- -- -- 	for t=1,T do
-- -- -- 	  mv,mi = torch.max(ex[i],2)
-- -- -- 	  exTensor[i] = allTwos - mi:t()
-- -- -- 	end
-- -- --       end
      exTensor = ex:gt(thr):reshape(N,T)
    end
  end
--   print(exTensor)
--   abort()
  return exTensor
end


function getFirstFrameLab(singleBatch)
  local labOne = torch.eye(maxTargets)*(-1)
  labOne = padTensor(labOne, maxDets, 2)
  labOne[labOne:eq(0)]=-2
  labOne = labOne:reshape(1,maxTargets*maxDets)
  
  if singleBatch then return dataToGPU(labOne) end

  firstFrameLab = torch.zeros(1,maxTargets*maxDets)
  for m=1,miniBatchSize do firstFrameLab=firstFrameLab:cat(labOne,1) end
  firstFrameLab=firstFrameLab:sub(2,-1)

  firstFrameLab = dataToGPU(firstFrameLab)
  
  return firstFrameLab
end

--------------------------------------------------------------------------
--- Returns a one-hot encoding of a label vector for one frame
-- @param lab			label (data association) vector for one frame
-- @param singleBatch	single- or mult-batch switch
function getOneHotLab(lab, singleBatch)
  
  local eye = torch.eye(nClasses)
  if singleBatch then
    
    local ind = lab:long():reshape(maxTargets)
--     print(ind)
    local labOne = eye:index(1,ind)
  --   labOne = pad
    return dataToGPU(labOne)
  else
    local labs = torch.zeros(1,nClasses)
    for mb=1,miniBatchSize do 
      local mbStartT = maxTargets * (mb-1)+1
      local mbEndT =   maxTargets * mb
      
      local ind = lab[{{mbStartT, mbEndT}}]:long():reshape(maxTargets)
      local labOne = eye:index(1,ind)
      
      labs = labs:cat(labOne, 1)
      
      
    end
    labs = labs:sub(2,-1)
    return dataToGPU(labs)
  end
end

--------------------------------------------------------------------------
--- TODO docs
-- @param lab			label (data association) vector for one frame
-- @param singleBatch	single- or mult-batch switch
function getOneHotLab2(lab, singleBatch)

end

function getOneHotLabAll(lab, singleBatch)
  if not singleBatch then error('multi-batch?') end
  
  local N,F = getDataSize(lab)
  local oneHot = torch.zeros(maxTargets,F,maxDets+1):int()
  for t=1,F do
    for i=1,N do
      oneHot[i][t][lab[i][t]]=1
    end    
  end
  return oneHot
end


function makePseudoProb(probVec, noise)
  local nClasses = probVec:size(1)
  local noise = noise or 0.001

  local minProb, maxProb = 1e-4, 1-1e-5
--   print(nClasses)
--   print(probVec)
--   print(torch.randn(nClasses)*0.001)
  probVec = probVec + torch.randn(nClasses):float()*noise
  probVec[probVec:le(0)] = minProb
  probVec[probVec:ge(1)] = maxProb

  local s = torch.sum(probVec)
  probVec = probVec / s
  return probVec
end

function oneHotPseudoProb(lab)
  local N,F,nClasses = getDataSize(lab)
  
  
  for t=1,F do
    for i=1,N do
      local probVec = lab[i][t]
      probVec = makePseudoProb(probVec)
      
      lab[i][t] = probVec
    end
  end
  
  return lab
  
end

--------------------------------------------------------------------------
--- Pad state tensors to accomodate 'false' state corresponding to false alarms
-- @param trTab		tracks table
-- @param detTab	detections table 
function addFalseState(trTab, detTab)
  if opt.max_nf == nil or opt.max_nf == 0 then return trTab end
  if opt.max_n == opt.max_m then return trTab end
  
  for k,v in pairs(trTab) do
    local newState = torch.zeros(1, opt.temp_win, opt.state_dim)  -- dummy row
    for mb=1,miniBatchSize do
      local mbStartT = opt.max_n * (mb-1)+1
      local mbEndT =   opt.max_n * mb
      local mbStartD = opt.max_m * (mb-1)+1
      local mbEndD =   opt.max_m * mb

	  -- pad tracks and detections
      newState = newState:cat(trTab[k][{{mbStartT, mbEndT},{},{}}], 1)
      newState = newState:cat(detTab[k][{{mbStartD+opt.max_n, mbEndD},{},{}}], 1)
    end
--     trTab[k] = trTab[k]:cat(detTab[k][{{opt.max_n+1,opt.max_m},{},{}}], 1)
    newState=newState:sub(2,-1)		-- remove first dummy row
    trTab[k] = newState:clone()
  end
  
  return trTab
end


--------------------------------------------------------------------------
--- Kalman Filters for state estimation
function createKalmanFilters()
  local KF = {}

  -- Number of mesurements and dimensions
  local M = opt.state_dim
  local D = M*2
  if sopt.use_KF~=0 and opt.state_dim<2 then
    sopt.use_KF=0
    print('WARNING! KF not implemented for dim<2')
  end

  -- KF Matrices
  local Ft    = torch.eye(D)
  local Ht    = torch.Tensor(M,D):zero()

  -- KF Variances  (location, velocity)
  -- local w1, w2 =.5, .01
  local w1 = torch.Tensor({.5, .5, .5, .5})
  local DT = 1 -- delta T
  local Wt    = torch.eye(D)*DT

  -- local Wt = torch.eye(D)*w1
  -- Wt[2][2] = w2 -- vx
  -- Wt[4][4] = w2 -- vy

  -- Measurement noise
  local v1 = torch.Tensor({.5, .5, 5, 5})*1
  local Vt    = torch.eye(M)

  -- print(sopt.use_KF)
  -- abort()
  if sopt.use_KF~=0 then
    print('USING KALMAN FILTER')
    
    for dd=1,M do Ft[dd*2-1][dd*2] = DT end
  --   print(Ft)
  --   abort()
  --   Ft[1][2] = DT -- State transition matrix
  --   Ft[3][4] = DT -- State transition matrix

  --   Ht[1][1] = 1 -- Measurement matrix
  --   Ht[2][3] = 1 -- Measurement matrix
    
    for dd=1,M do Ht[dd][dd*2-1] = 1 end
    
    for dd=1,M do Vt[dd][dd] = v1[dd] end

    
    for dd=1,D,2 do
      local a = (dd+1)/2
      print(a)
      Wt[dd][dd] = w1[a]*((DT*DT*DT)/3)
      Wt[dd][dd+1] = w1[a]*((DT*DT)/2)
      Wt[dd+1][dd] = w1[a]*((DT*DT)/2)
      Wt[a+1][a+1] = w1[a]*DT
      
    end

      
    for i=1,maxTargets do
      -- state    
      local X = torch.Tensor(D,1):zero()
      for dd=1,D,2 do
  --       print(dd)
	X[dd] = detections[i][1][(dd+1)/2] -- box x,y,w,h
  --     X[3] = detections[i][1][2] -- box y
      end
  --     print(X)
  --     abort()
      
      
      -- covariance (location, velocity)
  --     local s1,s2 = .5, .1
      local s = torch.Tensor({.5,.1,.5,.1,.5,.1,.5,.1})
      local S = torch.eye(D)
  --     S[2][2] = s2
  --     S[4][4] = s2
      for dd=1,D do
	S[dd][dd] = s[dd]
      end
  --     print(Ft)
  --     print(Ht)
  --     print(Wt)
  --     print(Vt)
  --     print(S)
  --     abort('KF')
      
      
      local KS = Kalman.create(string.format("ID %d",i), X, S)
      table.insert(KF, KS)    
    end
  end
  
  return KF, M, D, Ft, Wt, Ht, Vt
end

--------------------------------------------------------------------------
--- Print data association table
-- @param predDA		table containing log-probabilites (typically size N x F-1 x M+1
-- @param predLab		predicted labels (= max log-prob)
-- @param predEx		(optional) predicted existance of target
function printDA(predDA, predLab, predEx)
  local doColor = false
  -- do we want to color highlight the info/debug output?
  if opt~=nil and opt.colored_output ~= nil and opt.colored_output~=0 then doColor=true end
    
  local N,F,D = getDataSize(predDA)
  allMultiAss = 0
  
  -- calculate multi assignments
  for t=1,F do
    local multiAss = torch.zeros(nClasses)
    for i=1,N do
      multiAss[predLab[i][t+1]] = multiAss[predLab[i][t+1]]+1
    end
    allMultiAss = allMultiAss+torch.sum(multiAss:sub(1,maxDets):gt(1))
  end      
      
  if onNetwork() or maxTargets>5 then     
    print('Multi assignments: '.. allMultiAss)
    return allMultiAss
  end
  
  
  print(' ------- DATA  ASSOCIATION ------- ')
  local titlestr = string.format('%4s%4s%4s | ','frm','tar','ass')
  for d=1,maxDets do titlestr = titlestr..string.format('%6s%d','d',d) end
  titlestr = titlestr..string.format('%7s','miss')
  if D==maxDets+2 then titlestr = titlestr..string.format('%7s','trm') end
  titlestr = titlestr..string.format('%7s','Ex')
  print(titlestr)
  
  -- print predictions for first frame
  local t=1
  for i=1,1 do -- one target only (in the first frame, we don't have predictions anyway)
    local printline = ''
    printline = printline..(string.format('%4d%4d%4d | ',1,i,i))    
    for m=1,D do printline = printline .. (string.format('%7s','-.---')) end
    if exVar or predEx~=nil then printline = printline .. (string.format('%7s','-.---')) end
    print(printline)
  end
  
  
  local col = sys.COLORS
  if not doColor then 
    col={}
    col.cyan, col.red, col.white = '','','' 
  end
  
  for t=1,math.min(F,5) do
    print('--------------')
    for i=1,N do      
      local printline = ''
      local assID = predLab[i][t+1] -- assigned ID (max. prob)
      if assID == nClasses then assID = 0 end -- print 0's for missed
      printline = printline..(string.format('%4d%4d%4d | ',t+1,i,assID))  
       for m=1,maxDets+1 do
	 local cstr = ''
	 if m==predLab[i][t+1] then cstr = col.cyan end
	 printline = printline .. cstr .. (string.format('%7.3f',torch.exp(predDA[i][t][m]))) .. col.white	 
-- 	 if m==maxDets+1 then printline=printline..'\n' end
       end
       if D==maxDets+2 then
	 local cstr = ''
	 local m=D
	 if m==predLab[i][t+1] then cstr = col.cyan end
	 printline = printline .. cstr .. (string.format('%7.3f',torch.exp(predDA[i][t][m]))) .. col.white	 	 
       end
       
       if exVar or predEx~=nil then
	if predEx[i][t][1]>0.5 then 
	  printline = printline .. col.red .. (string.format('%7.3f',predEx[i][t][1])) .. col.white
	else
	  printline = printline .. (string.format('%7.3f',predEx[i][t][1]))
	end	
       end
       
       print(printline)
    end    
  end
  print('Multi assignments: '.. allMultiAss)
  return allMultiAss
end

function getStateRange()
  local dRange = torch.zeros(3,opt.state_dim)
  for d=1,opt.state_dim do
    dRange[1][d]=-.5
    dRange[2][d]=.5
    dRange[3][d]=1  
  end  
  return dRange
end

--------------------------------------------------------------------------
--- Fill in all non-existing states with random values
-- @param tab		table with states (detections or GT)
-- @param ex		table with existance indicators (GT only)
function injectRandomClutter(tab, ex)
  local dRange = getStateRange()
  
  -- detections
  if ex == nil then
    for k,v in pairs(tab) do
      for t=1,opt.temp_win do
	for d=1,stateDim do
	  for det=1,maxDets*miniBatchSize do  
	    if tab[k][det][t][d] == 0 then
	      tab[k][det][t][d] = torch.rand(1):squeeze() * dRange[3][d] + dRange[1][d]
	    end	
	  end
	end
      end
    end  
  else
    -- targets
    for k,v in pairs(tab) do
      for t=1,opt.temp_win do
	for d=1,stateDim do
	  for tar=1,maxTargets do  
	    if ex[k][tar][t] == 0 then
	      tab[k][tar][t][d] = torch.rand(1):squeeze() * dRange[3][d] + dRange[1][d]
	    end	
	  end
	end
      end
    end      
  end
  return tab
  
end

--------------------------------------------------------------------------
--- TODO DOCS
function removeInexistantDetections(dets, ex, lab)
  local dRange = getStateRange() 

  -- detections
  for k,v in pairs(dets) do
    for t=1,opt.temp_win do
      for d=1,stateDim do
	for mb = 1,opt.mini_batch_size do
	  local mbStart = opt.max_n * (mb-1)+1
	  local mbEnd =   opt.max_n * mb	
	  
	  local mbStartD = opt.max_m * (mb-1)+1
	  local mbEndD =   opt.max_m * mb	
	  
	  for tar=1,maxTargets do
	    local newI = mbStart+tar-1
	    if ex[k][newI][t] == 0 then
	      local thisLab = lab[k][newI][t]
	      if thisLab <= maxDets then
		dets[k][thisLab+mbStartD-1][t][d] = torch.rand(1):squeeze() * dRange[3][d] + dRange[1][d]
	      end
	    end	
	  end
	end
      end
    end
  end      
  return dets
  
end


--------------------------------------------------------------------------
--- Smoothly (linearly) interpolate existance label between 1 and 0
-- @param ExTab		exlabels table
-- @param delay		number of frames to interpolate
function smoothTerminationLabel(ExTab, delay)
  delay = delay or 4
  for k,v in pairs(ExTab) do
    ExTab[k] = v:float():clone()
    for i=1,maxTargets do
      for t=2,opt.temp_win do
	local exTprev = ExTab[k][i][t-1]
	local exT = ExTab[k][i][t]
	if exT==0 and exTprev ~= 0 then	 
	  ExTab[k][i][t] = math.max(0,exTprev-1/delay)
	end
      end
    end
  end
  return ExTab
end


function logRevision()
  print('Current Revision:')
  hgrevStr = 'hg identify -n; hg identify'
  os.execute(hgrevStr)
  hgrevStr = string.format('hg identify -n > %srev.txt', outDir)
  os.execute(hgrevStr)
end


function toBits(num,bits)
    -- returns a table of bits, most significant first.
    bits = bits or select(2,math.frexp(num))
    local t={} -- will contain the bits        
    for b=bits,1,-1 do
	t[b]=math.fmod(num,2)
	num=(num-t[b])/2
    end
    return t
end
  

function onNetwork()
  if lfs.attributes('/home/h3/amilan/','mode') then
    return true
  end

  return false
end


function getPWD(opt, input, detections, t, detexlabels)
  local loctimer=torch.Timer()
  local missThr = 1;  
  local distNorm = 2; -- Euclidean
  local maxDim = 2
  local pwdDim = math.min(stateDim, maxDim)


  
  local allDistBatches = torch.zeros(miniBatchSize, maxTargets*maxDets):float()
  if opt.pwd_mode == 1 then allDistBatches = torch.zeros(miniBatchSize, maxTargets*maxDets*pwdDim):float() end
  

  for mb = 1,opt.mini_batch_size do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb

    local mbStartD = opt.max_m * (mb-1)+1
    local mbEndD =   opt.max_m * mb

    
    local det_x =	detections[{{mbStartD, mbEndD},{t+1},{1,pwdDim}}]:clone():reshape(maxDets,pwdDim)
--     print(input)
--     print(mbStart, mbEnd)
    local inpPred = input[mb]:reshape(maxTargets, stateDim):narrow(2,1,pwdDim)
--     print(t+1)
--     print(inpPred)
--     print(det_x)
--     print(detexlabels)

      

    local loctimer1=torch.Timer()
    
    local allDist = torch.zeros(maxTargets,maxDets):float()
    if opt.pwd_mode == 1 then allDist = torch.zeros(maxTargets, maxDets, pwdDim) end
    
    for tar=1,maxTargets do    
      for det=1,maxDets do
	if TESTING and detexlabels~=nil and detexlabels[det][t+1]==0 then
	  dist = missThr*2
	  if opt.pwd_mode == 1 then allDist[{{tar},{det},{}}] = dist
	  else allDist[{{tar},{det}}] = dist
	  end
	else

	  if opt.pwd_mode ==1 then
	    for dim=1,pwdDim do
	      dist = torch.abs(inpPred[tar][dim]-det_x[det][dim])
	      allDist[tar][det][dim] = dist
	    end
	  
	  else
	    
	    if opt.pwd_mode == 2 then
	      error('overlap needs fixing')
	      -- overlap 	    
	      dist = 1-boxIoU(inpPred[tar]+0.5,det_x[det]+0.5)
	      allDist[tar][det] = dist

	    else -- default: norm
	      dist = torch.dist(inpPred[tar], det_x[det], distNorm)
	      allDist[tar][det] = dist
	    end
	  end
	end
      end    
    end
    
--     print(allDist)
--     if opt.profiler ~= 0 then  profUpdate('loop PWD', loctimer1:time().real) end 
--     local loctimer1=torch.Timer()
--     local fastAllDist = doFastPWD(inpPred, det_x)
--     if opt.profiler ~= 0 then  profUpdate('fast PWD', loctimer1:time().real) end 
--     print(fastAllDist)
--     abort()
    
    
    if opt.pwd_mode == 1 then
      allDistBatches[mb] = allDist:reshape(1,maxTargets*maxDets*pwdDim)
    else
      allDistBatches[mb] = allDist:reshape(1,maxTargets*maxDets)
    end
  end
  
--   print(allDistBatches:reshape(maxTargets,maxDets,stateDim))
--   sleep(2)	

  if opt.profiler ~= 0 then  profUpdate('getPWD', loctimer:time().real) end 
  return allDistBatches
  
end


function getDAError(predDA, labels)
  local loctimer=torch.Timer()
  local loclabs = labels:narrow(2,2,opt.temp_win-1):int()
  local predLabRaw = getLabelsFromLL(predDA, false)
  -- pad 1,2,3...
--   local nRows = maxTargets
-- --   print(labels)
-- --   print(predLabRaw)
--   local oneBatch = torch.linspace(1,nRows,nRows):int():reshape(nRows,1)
--   local pad=oneBatch:clone()
--   for m=2,miniBatchSize do pad=pad:cat(oneBatch,1) end
-- --   print(pad)
--   predLabRaw = torch.cat(pad,predLabRaw,2)

  local misDA = torch.abs(predLabRaw-loclabs)
  misDA[misDA:ne(0)]=1
  
  local N,F = getDataSize(loclabs)
  local errPercent = torch.sum(misDA) / (N*F) *100

  if opt.profiler ~= 0 then  profUpdate(debug.getinfo(1,"n").name, loctimer:time().real) end 
  
--   print(predDA)
--     print(predLabRaw)
--   print(loclabs)
--   print(misDA)
--   print(errPercent)
--   abort()
--   
  print(string.format('Mispredicted labels: %d = %.2f %%',torch.sum(misDA), errPercent))
  return errPercent, misDA
end

function getDAErrorHUN(predDA, labels)
  local predLabRaw = getLabelsFromLL(predDA, false)

  local misDA = torch.abs(predLabRaw-labels)
  misDA[misDA:ne(0)]=1
  
  local N,F = getDataSize(labels)
  local errPercent = torch.sum(misDA) / (N*F) *100
  print(string.format('Mispredicted labels: %d = %.2f %%',torch.sum(misDA), errPercent))
  return errPercent, misDA
end

function doFastPWD(inpPred, det_x)
  -- turns out to be slower than loops
  local allDist = torch.zeros(maxTargets, maxDets):float()
  
  -- dim 1
  local reshInp = inpPred:narrow(2,1,1):reshape(maxTargets,1):expand(maxTargets,maxDets,pwdDim)
  local reshDet = det_x:narrow(2,1,1):reshape(1,maxDets):expand(maxTargets,maxDets)
  
  local d1 = torch.abs(reshInp - reshDet):reshape(maxTargets, maxDets, 1)
  
  return d1
  
end



function getFillMatrix(batchmode, missThr, dummyNoise, pwdDim)  
  local mb = 1
  if batchmode == nil then batchmode = true end
  
  if batchmode then mb = miniBatchSize end
  
  local fillMatrix = torch.ones(mb*maxTargets,maxTargets) * missThr
  if dummyNoise ~= 0 then
    fillMatrix = fillMatrix + torch.rand(mb*maxTargets,maxTargets) * dummyNoise
  end
  
  if pwdDim>1 then 
    fillMatrix = fillMatrix:reshape(mb*maxTargets,maxTargets,1):expand(mb*maxTargets,maxTargets,pwdDim) 
  end
  
  return fillMatrix:float()
end

function getPWDHUN(pwdMode, nClasses, pwdDim, tracks, detections, missThr, dummyNoise)
  local distNorm = 2; -- Euclidean    
  local allDistBatches = torch.zeros(miniBatchSize, maxTargets*nClasses*pwdDim):float()
  
  for mb = 1,opt.mini_batch_size do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb

    local mbStartD = opt.max_m * (mb-1)+1
    local mbEndD =   opt.max_m * mb
    
    local allDist = torch.zeros(maxTargets,maxDets):float()
    if pwdMode > 0 then allDist = torch.zeros(maxTargets, maxDets, pwdDim):float() end
    
    local det_x = detections[{{mbStartD, mbEndD},{}}]
    local inpPred = tracks[{{mbStart, mbEnd},{}}]
    
--       print(det_x)
--       print(inpPred)
--       abort()
    
    for tar=1,maxTargets do    
      for det=1,maxDets do
	

	if pwdMode ==1 then
	  for dim=1,pwdDim do
	    dist = torch.abs(inpPred[tar][dim]-det_x[det][dim])
	    allDist[tar][det][dim] = dist
	  end	
	elseif pwd_mode == 2 then
	  for dim=1,pwdDim do
	    dist = inpPred[tar][dim]-det_x[det][dim]
	    allDist[tar][det][dim] = dist
	  end	
	else -- default: norm
	  dist = torch.dist(inpPred[tar], det_x[det], distNorm)
	  allDist[tar][det] = dist
-- 	  
-- 	  if pwdMode == 2 then
-- 	    error('overlap needs fixing')
-- 	    -- overlap 	    
-- 	    dist = 1-boxIoU(inpPred[tar]+0.5,det_x[det]+0.5)
-- 	    allDist[tar][det] = dist
-- 
-- 	  
	end
	
      end
    end
    if pwdMode >0 then      
--       print(allDist)
--       print(getFillMatrix(false):reshape(maxTargets,maxDets,1):expand(maxTargets,maxDets,pwdDim))
--       abort()
      allDist = allDist:cat(getFillMatrix(false, missThr, dummyNoise, pwdDim),2)
      allDistBatches[mb] = allDist:reshape(1,maxTargets*nClasses*pwdDim)
    else
      allDist = allDist:cat(getFillMatrix(false, missThr, dummyNoise, pwdDim),2)
      allDistBatches[mb] = allDist:reshape(1,maxTargets*nClasses)
    end      
      
  end    
  
  return allDistBatches
end

--------------------------------------------------------------------------
--- Make directory if does not yet exist
-- @param dir   path to dir to create
function mkdirP(dir)
  if not lfs.attributes(dir) then 
    lfs.mkdir(dir)
    pm(string.format('Directory %s created',dir)) 
  end
end


function createAuxDirs()
  local rootDir = getRNNTrackerRoot()
  mkdirP(rootDir..'/bin')
  mkdirP(rootDir..'/tmp')
  mkdirP(rootDir..'/out')
  mkdirP(rootDir..'/config')
  mkdirP(rootDir..'/graph')
end