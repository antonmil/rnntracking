-----------------------
--- Module including many miscellaneous utility functions
--
-- @module types

require 'gnuplot'	-- plotting stuff
require 'lfs' 		-- luaFileSystem, check for dir, etc...
require 'util.csv'	-- reading CSV files
require 'image'		-- image load
require 'util.plot'	-- plotting utils
require 'util.io'	-- reading / writing data
require 'util.matlab'	-- matlab like tools, union, intersect, find,...


--------------------------------------------------------------------------
--- Utility function. TODO: move away to some utils file?
-- takes a list of tensors and returns a list of cloned tensors
-- @author Andrej Karpathy
function clone_list(tensor_list, zero_too)        local timer=torch.Timer()
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    profUpdate(debug.getinfo(1,"n").name, timer:time().real) return out
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
--- Abort the execution of the program
function abort()
  os.exit(0)
end


--------------------------------------------------------------------------
--- Orders detections in one frame according to their distance to tracks.
function orderDets(detections, tracks, daMode) local timer=torch.Timer()
  daMode = daMode or 2 -- 1=greedy, 2=hungarian
--   daMode = 2
  
  local Ndet,Fdet = getDataSize(detections)
  local N,F = getDataSize(tracks)
  if detections:nDimension() > 2 then
    detections = detections:narrow(3,1,1):reshape(Ndet,Fdet) -- use X dim only
  end
  if tracks:nDimension() > 2 then tracks = tracks:narrow(3,1,1):reshape(N,F) end

  local detOrdered = detections:clone()
  local ndet = detections:size(1)
  local da = torch.zeros(ndet)
  
  -- GREEDY DATA ASSOCIATION
  if daMode == 1 then
    for t=1,opt.temp_win do
--     for t=1,1 do
      for i = 1,tracks:size(1) do
	-- dist of track i to all det
	cloneTrack = torch.repeatTensor(tracks[{{i},{t}}],ndet,1)
	distVec = torch.abs(cloneTrack-detections[{{},{t}}])
	sDist,cDet = torch.min(distVec,1) -- find closest detection to track i
	sDist=sDist[1][1]; cDet=cDet[1][1]
	detOrdered[i][t] = detections[cDet][t]
	da[cDet]=i
	
      end
    --   print("Smallest dist: "..sDist..". Nearest Det: "..cDet)    
    end
  end
  
  if daMode == 2 then
    t=1
    -- create cost matrix
    cost = torch.Tensor(tracks:size(1),ndet)
    for i=1,tracks:size(1) do
	cloneTrack = torch.repeatTensor(tracks[{{i},{t}}],ndet,1)
	distVec = torch.abs(cloneTrack-detections[{{},{t}}])
	cost[i] = distVec:t()
    end
    -- we need 1 / dist as cost? NO
  --   cost = torch.cdiv(torch.ones(cost:size(1), cost:size(2)), cost)
    
    cost[cost:gt(0.03)]=10		-- jump to max if over threshold
    cost[cost:gt(10)]=10		-- avoid Inf
  --   print(cost)
    
--     csvWrite('tmp/cost.txt',cost:t())	-- write cost matrix
--     os.execute("python hung.py")	-- do python munkres
--     ass = csvRead('tmp/ass.txt') 	-- read result
--     ass = torch.Tensor(ass)+1	-- convert back to tensor
--     print('---------------------')
--     print(detections)
--     print(tracks)
--     print(cost)
--     print(ass)
    ass = hungarian(cost)
--     print(ass)
    
    for i=1,ndet do
      detOrdered[i] = detections[ass[i][2]][t]
      da[i]=ass[i][2]
    end
  --   print(detOrdered)
  --   sleep(1)
  --   print(ass)
  --   abort()
    
  end -- daModa
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return detOrdered, da
end




--------------------------------------------------------------------------
--[[ getTrack
  Retruns box coordinates over the entire track as well as tstart and tend

  in:	gt	Table with GT info
	qid	Query track ID 
  out:	trackT	a Tx4 Tensor with bbox coordinates [bx,by,bw,bh]
	ts	track start frame
	te	track end frame

]]--
--------------------------------------------------------------------------
function getTrack(gt,qid) local timer=torch.Timer()
  
  local track = {}
  local ts=1e10
  local te=-1e10
  for fr,id in pairs(gt) do
    local inframe = id
    if id[qid] ~= nil then
--       print(fr,id)
      table.insert(track,id[qid])
      if fr < ts then ts = fr end -- probably not the smartest way
      if fr > te then te = fr end --   to determine s and e
    end
  end
  
  -- convert to Tensor
  local trLength = te-ts+1    
  local trackT = torch.Tensor(0)
  if trLength > 0 then
    trackT = torch.Tensor(trLength,4)
    for t,state in pairs(track) do
      for s=1,4 do
	trackT[t][s] = state[1][s]
      end
    end
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return trackT, ts, te
end

--------------------------------------------------------------------------
--- Reshuffle detections to remove IDs
-- @param detections	The data tensor
function reshuffleDets(detections) local timer=torch.Timer()
  local scrdet = detections:clone()
  local N,F = getDataSize(detections)
  local scrindex = torch.IntTensor(detections:size(1), detections:size(2))
  for t=1,F do
    scrindex[{{},{t}}] = torch.randperm(N)
--     scrdet[{{},{t}}] = detections[{},{t}}]
    
    for i = 1,N do    
--       print(t,i)
--       print(scrdet[{{i},{t}}],detections[{{scrindex[i]},{t}}])
--       print(scrdet[{{i},{t}}])
      
      scrdet[i][t] = detections[scrindex[i][t]][t]
    end
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return scrdet, scrindex
end

--------------------------------------------------------------------------
--- Reshuffle tracks
-- @param data	The data tensor
function reshuffleTracks(data) local timer=torch.Timer()
  local scrdata = data:clone()
  local N,F,D = getDataSize(data)
  local scrindex = torch.IntTensor(N)
  scrindex = torch.randperm(N)
  for i=1,N do
    scrdata[{{i},{},{}}] = data[{{scrindex[i]},{},{}}]
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return scrdata, scrindex
end

--------------------------------------------------------------------------
--- Get directory where sequences are stored.
function getDataDir() local timer=torch.Timer()
  local dataDir
  if lfs.attributes('/media/sf_vmex','mode') then -- virtual machine
    dataDir = '/media/sf_vmex/2DMOT2015/data/'
  elseif lfs.attributes('/home/amilan/research/projects/bmtt-data/','mode') then -- PC in office Adelaide
    dataDir = '/home/amilan/research/projects/bmtt-data/'
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return dataDir  
    
end

--------------------------------------------------------------------------
--- Get ground truth of a specific sequence as a table.
-- @param seqName	The name of the sequence.
function getGT(seqName) local timer=torch.Timer()
  datafile = getDataDir() .. seqName .. "/gt/gt.txt"
  if lfs.attributes(datafile) then 
--     print("GT file ".. datafile)
  else
    error("Error: GT file ".. datafile .." does not exist")
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return readTXT(datafile, 1) -- param #2 = GT
end

--------------------------------------------------------------------------
--- Get result of a specific sequence as a table.
-- @param seqName	The name of the sequence.
function getRes(seqName) local timer=torch.Timer()
  datafile = "out/".. seqName .. ".txt"
  if lfs.attributes(datafile) then 
--     print("GT file ".. datafile)
  else
    error("Error: Results file ".. datafile .." does not exist")
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return readTXT(datafile, 1) -- param #2 = GT
end

--------------------------------------------------------------------------
--- Get ground truth of a sequence as a tensor
-- @param seqName 	The name of the sequence.
-- WARNING Only keeps tracks present in first frame (ts)
-- TODO: fix
-- DEPRECATED 
function getGTTracksOLD(seqName)
  ts = ts or 1 	-- default starting frame
--   F = F or 1e10	-- default sequence duration
  local nDim = 4		-- number of dimensions
  
  gt = getGT(seqName)	-- get raw GT table
  
  
  local imgHeight, imgWidth = getImgRes(seqName) -- required for normalization
--   F = math.min(tabLen(gt)-ts+1,F)
  F = tabLen(gt)
  
  -- which tracks (IDs) are in first frame
  local orderedKeys = {}
  for k in pairs(gt[ts]) do table.insert(orderedKeys, k) end
  table.sort(orderedKeys)
  
  local N = tabLen(orderedKeys)		-- number of tracks
  local tracks = torch.ones(N, F, nDim):fill(0)
  
  -- Now create a FxN tensor with states
  for t=1,F do
    local cnt=0
    for i = 1, N do
	cnt=cnt+1
	local gtID, gtBBox = orderedKeys[i], gt[t+ts-1][ orderedKeys[i] ]
	if gtBBox then 
	  for dim = 1,nDim do
-- 	    tracks[cnt][t] = getFoot(gtBBox)[1][1] / imgWidth 	    
	    tracks[cnt][t][dim]=gtBBox[1][dim] / imgWidth 	-- normalize to [0,1]
	  end
	end
    end
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tracks
end

--------------------------------------------------------------------------
--- Get ground truth of a sequence as a tensor. NOTE All tracks are retrieved.
-- @param seqName 	The name of the sequence.
function getGTTracks(seqName) local timer=torch.Timer()
  ts = ts or 1 	-- default starting frame
--   F = F or 1e10	-- default sequence duration
  local nDim = 4		-- number of dimensions
  
  gt = getGT(seqName)	-- get raw GT table
  
  
  local imgHeight, imgWidth = getImgRes(seqName) -- required for normalization
--   F = math.min(tabLen(gt)-ts+1,F)
  local tL,minKey, maxKey = tabLen(gt)
  local F = maxKey
--   print(tL, minKey, maxKey)
--   F=50;
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
  tracks = cleanDataTensor(tracks)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tracks
end


--------------------------------------------------------------------------
--- Get results of a sequence as a tensor. NOTE All tracks are retrieved.
-- @param seqName 	The name of the sequence.
function getResTracks(seqName) local timer=torch.Timer()
  ts = ts or 1 	-- default starting frame
  local nDim = 4		-- number of dimensions
  
  res = getRes(seqName)	-- get raw GT table
  
  local imgHeight, imgWidth = getImgRes(seqName) -- required for normalization
  imgWidth = 1
  local tL,minKey, maxKey = tabLen(res)
  
  if tL == 0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return torch.zeros(0,0,nDim) end
  local F = maxKey
  local N = 255
  
  local tracks = torch.zeros(N, F, nDim)
  
--   print(res)
--   print(tL)
--   print(F)
--   print(N)
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
--   print(tracks)
  tracks = cleanDataTensor(tracks)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tracks
end

--------------------------------------------------------------------------
--- Crop GT or detection tensors to a specific time window.
-- @param data	The data tensor.
-- @param ts	First frame.
-- @param te	Last frame.
function selectFrames(data, ts, te, clean) local timer=torch.Timer()
--   if clean == nil then clean = true end
  local N,F,D = getDataSize(data)		-- data size
  assert(ts<=te,"Start point must be <= end point")
  if ts<1 or te>F then print("Warning! Frame selection outside bounds") end
--   if ts>F then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return torch.zeros(N,te-ts+1,D) end
  
  ts=math.max(1,ts);  te=math.min(F,te) -- make sure we are within bounds
  
  data=data:narrow(2,ts,te-ts+1) 	-- perform selection
  
  if clean == nil or clean == true then
    data = cleanDataTensor(data)		-- clean zero tracks TODO
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data    
end

--------------------------------------------------------------------------
--- Remove all-zero tracks (rows).
-- @param data	The data Tensor N x F x Dim.
function cleanDataTensor(data) local timer=torch.Timer()
  local N,F = getDataSize(data)
  local exTar = torch.ne(torch.sum(data:narrow(3,1,1),2),0):reshape(N) -- binary mask
  local nTar = torch.sum(exTar) -- how many true tracks
  
  
  local newData = torch.Tensor(nTar,F,data:size(3))
  local cnt = 0
  for i=1,N do
    if exTar[i] > 0 then
      cnt=cnt+1
      newData[{{cnt},{}}]=data[{{i},{}}]
    end
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return newData
end

--------------------------------------------------------------------------
--- Get detections of a specific sequence as a table.
-- @param seqName	The name of the sequence.
-- @profUpdate(debug.getinfo(1,"n").name, timer:time().real) return 		A table containing the detection bounding boxes
function getDetections(seqName) local timer=torch.Timer()
  datafile = getDataDir() .. seqName .. "/det/det.txt"
  if lfs.attributes(datafile) then 
--     print("Detections file ".. datafile)
  else
    error("Error: Detections file ".. datafile .." does not exist")
  end     
  return readTXT(datafile, 2) -- param #2 = detections
end

--------------------------------------------------------------------------
--- Get detections of a specific sequence as a tensor.
-- @param seqName	The name of the sequence.
-- @return 		A Nd x F x 4 Tensor
-- @see getDetections
-- @see getDetectionsVec
function getDetTensor(seqName) local timer=torch.Timer()
  local detTab = getDetections(seqName)
  local imgHeight, imgWidth = getImgRes(seqName)
  F = tabLen(detTab)
  local nDim = 4 	-- bounding box
  
  local maxDetPerFrame = 0
  for t=1,F do
    if detTab[t] and tabLen(detTab[t]) > maxDetPerFrame then maxDetPerFrame = tabLen(detTab[t]) end
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
  
  -- pad detections at end if necessary
  local Ngt,Fgt,Dgt = getDataSize(getGTTracks(seqName))
  if Fgt>F then
    detections=detections:cat(torch.zeros(maxDetPerFrame,Fgt-F,nDim),2)
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return detections
end



-- special ID for false alarm
function getFALabel() local timer=torch.Timer()
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return -1
end

--------------------------------------------------------------------------
--- Selects a specific state dimension from a NxFxD tensor.
-- @return Only one dimension, i.e. an NxF tensor
function selectStateDim(data, dim) local timer=torch.Timer()
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data:narrow(3,dim,1):reshape(data:size(1),data:size(2))
end

--------------------------------------------------------------------------
--- Naively associate detections.
--  For each detection find closest track and if it is below a threshold, take it.
--
-- @param detections	FdxNd tensor: current detections in arbitrary order
-- @param tracks	FxN tensor:   tracks to associate detections to
-- @param thr 		threshold to consider inlier
-- @return 		FdxNd int tensor with det-to-track IDs
function associateDetections(detections, tracks, thr) local timer=torch.Timer()
--   detections = selectStateDim(detections,1) 	-- select x as primary dim
--   tracks = selectStateDim(tracks,1)
  
  local Ndet,Fdet, stateDim = getDataSize(detections)	-- detection tensor size
  local N,F = getDataSize(tracks)		-- tracks tensor size
  local da = torch.IntTensor(Ndet,Fdet):fill(0)	-- new tensor w/ assc. detIDs
  da[torch.gt(selectStateDim(detections,1),0)] = getFALabel()	-- fill in all as false alarm
  
  thr = thr or 0.03
  if stateDim == 4 then thr = 0.5 end
  
  -- loop through all detections
  for t=1,math.min(Fdet,F) do
    for d=1,Ndet do      
      -- find closest track to this particular detection
      if detections[d][t][1] > 0 then -- ignore dummy detections
	local sDist, cTr = findClosestTrack(detections[d][t], tracks[{{},{t}}])
	
	if sDist < thr then	-- associate if below threshold      
	  da[d][t] = cTr
	end
      end
    end
  end  
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return da
  
end


--------------------------------------------------------------------------
--- Intersection over union of two boxes.
-- The boxes are given as 1D-Tensors of size 4 with x,y,w,h
function boxIoU(box1, box2) local timer=torch.Timer()
  local ax1=box1[1]; local ax2=ax1+box1[3]; local ay1=box1[2]; local ay2=ay1+box1[4]
  local bx1=box2[1]; local bx2=bx1+box2[3]; local by1=box2[2]; local by2=by1+box2[4]  
  
  local hor = math.max(0,math.min(ax2, bx2) - math.max(ax1,bx1)) -- horiz. intersection
  
  local ver = 0
  if hor>0 then
    ver = math.max(0,math.min(ay2, by2) - math.max(ay1,by1)) -- vert. intersection
  end
  local isect = hor*ver
  local union = box1[3]*box1[4] + box2[3]*box2[4] - isect
  if union==0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return 0 end -- empty boxes special case
  
  assert(union>0,'Something wrong with union')
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return isect / union
  
end

--------------------------------------------------------------------------
--- Find closest track to one particular detection. Depending on the 
-- state dimension, the distance is either absolute distance (1D),
-- Euclidean distance (2D), or 1-IoU (4D)
-- @param det		Detection (A 1D, 2D, or 4D vector)
-- @param tracks	All tracks in one frame (Nx1xD)
-- @return 		Closest distance
-- @return 		Closest tack ID
function findClosestTrack(det, tracks) local timer=torch.Timer()
  -- Find out state dimension
  local N, F, stateDim = getDataSize(tracks)
  assert(stateDim == det:size(1),'state dimensions do not agree')

  
  local sDist={};  local cTr = {}
  if stateDim == 1 then		-- Absolute distance
    tracks = tracks:reshape(N) -- make an N-D vector
    -- clone provided detection N times to match tracks vector
    local cloneDet = torch.repeatTensor(det,N,1)
    local distVec = torch.abs(cloneDet-tracks)
    sDist,cTr = torch.min(distVec,1) -- closest track (dist, ID)
    sDist=sDist:squeeze(); cTr = cTr:squeeze() -- squeeze for dx1-size to d-size     
    
  elseif  stateDim == 4 then
    sDist = 1 -- largest distance    
    for i=1,N do
      if det[1] > 0 and tracks[i]:squeeze()[1] > 0 then -- ignore dummies
	ioud = 1-boxIoU(det,tracks[i]:squeeze())
	if ioud < sDist then
	  sDist = ioud
	  cTr = i
	end    
      end
    end

  else
    error('distance computation for state dim '..stateDim..' not implemented')
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return sDist, cTr
end



--------------------------------------------------------------------------
--- Compute distance between two states
-- The actual computation depends on the state dimension.
-- 1D and 2D states invoke Euclidean distance
-- 4D state assumes two bounding boxes and computes the IoU score
-- @param state1
-- @param state2
function computeDistance(state1, state2) local timer=torch.Timer()
  error('computeDistance not implemented')
  
end

--------------------------------------------------------------------------
--- Pad tracks or detections to a consistent size
-- @param tracks	The tracks tensor
-- @param detections	The detections tensor
-- @param da		The det-to-ID association tensor
-- @param maxTracks	Option: The max number of tracks / classes to pad to
function unifyData(tracks, detections, da, maxTracks) local timer=torch.Timer()
  local N,F,D = getDataSize(tracks)
  local Ndet,Fdet,Ddet = getDataSize(detections)  
--   assert(F==Fdet,"Hmm... number of frames of GT and Det not equal!")
  -- pad frames
--   if Fdet > F then tracks = torch.cat(tracks, torch.zeros(N, Fdet-F, D), 2) end
  assert(F>=Fdet,"Hmm... more detection frames than ground truth. Is that possible?")
  if F > Fdet then 
    detections = torch.cat(detections, torch.zeros(Ndet,F-Fdet,Ddet), 2)
    
    da = torch.cat(da, torch.IntTensor(Ndet,F-Fdet):fill(0), 2)
    
    -- fill in false alarms
    for i=1,Ndet do
      for t=Fdet+1,F do
	if tracks[i][t][1]>0 then da[i][t] = getFALabel() end
      end
    end
    Ndet,Fdet,Ddet = getDataSize(detections)
  end
  
  if Ndet > N then    -- too many detections, pad tracks
    tracks = torch.cat(tracks, torch.zeros(Ndet-N,F,D),1)
    
  elseif N > Ndet then 	-- too many tracks, pad detections and da
    detections = torch.cat(detections, torch.zeros(N-Ndet,Fdet,Ddet), 1)
    
    da = torch.cat(da, torch.IntTensor(N-Ndet,F):fill(0), 1)
    
    -- fill in false alarms
    for i=Ndet+1,N do
      for t=1,F do
	if tracks[i][t][1]>0 then da[i][t] = getFALabel() end
      end
    end
  end

  
  -- optional: define fixed max number of tracks
  if maxTracks then
    N,F = getDataSize(tracks)
    Ndet,Fdet = getDataSize(detections)
    
    if maxTracks > N then 
      tracks = torch.cat(tracks, torch.zeros(maxTracks-N,F,D),1)
      N,F,D = getDataSize(tracks)
    end
    if maxTracks > Ndet then
      detections = torch.cat(detections, torch.zeros(maxTracks-Ndet,F,Ddet), 1)      
      da = torch.cat(da, torch.IntTensor(maxTracks-Ndet,F):fill(0), 1)
    end
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tracks, detections, da
  
end

--------------------------------------------------------------------------
--- Randomly remove detections to simulate detector failure.
-- TODO Add doc
function removeDetections(detections, failureRate) local timer=torch.Timer()
  failureRate = failureRate or 0.2 -- percent of detector failure

  local ldet = detections:clone()
--   local N=detections:size(2)
--   local F=detections:size(1)
  local N,F,D = getDataSize(detections)
  
  local mask = torch.rand(F,N,D)
  mask = torch.le(mask,failureRate) -- binary mask to remove
  ldet:maskedFill(mask,0)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return ldet

end

--------------------------------------------------------------------------
--- Make artificial detections more realistic. Add noise, remove dets, add false alarms
-- @param detections	The detections tensor
-- @param noiseVar	The noise variance multiplier
-- @param failureRate	The failure rate (0,1)
-- @param falseRate	The false alarm rate (empty spots in tensor are filled with this probability)
function perturbDetections(detections, noiseVar, failureRate, falseRate) local timer=torch.Timer()
  
  noiseVar = noiseVar or 0.01
  failureRate = failureRate or 0.2
  falseRate = falseRate or 0.1

  local N,F,D = getDataSize(detections)
  
  local zeroMask = torch.le(detections, 0)	-- find inexistent detections
  detections = detections + torch.randn(N,F,D) * noiseVar
  detections[zeroMask] = 0
  detections = removeDetections(detections, failureRate)
  
  -- this is most likely inefficient
  for t=1,F do
    for d=1,N do
      if detections[d][t][1] == 0 then
	if math.random() < falseRate then
	  detections[{{d},{t}}] = torch.rand(D)
	end
      end
    end
  end
  
  
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return detections
end

--------------------------------------------------------------------------
--- Returns number of rows, number of columns (optional number of dim3) of a 2D Tensor
function getDataSize(data) local timer=torch.Timer()
  local R=0
  local C=0
  local D=0
  
  if data:nDimension()>=1 then R=data:size(1) end
  if data:nDimension()>=2 then C=data:size(2) end  
  if data:nDimension()>=3 then D=data:size(3) end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return R,C,D
end

--------------------------------------------------------------------------
--- Get the bottom center point of a box
-- @param bbox	A 1x4 tensor describing the bbox as [bx,by,bw,bh]
-- @return A 1x2 tensor giving the x,y position of the feet
function getFoot(bbox)  local timer=torch.Timer()
  pos = torch.Tensor(1,2):fill(0)
  pos[1][1] = bbox[1][1] + bbox[1][3]/2
  pos[1][2] = bbox[1][2] + bbox[1][4]
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return pos
end

--------------------------------------------------------------------------
--- Get the center point of a box
-- @param bbox	A 1x4 tensor describing the bbox as [bx,by,bw,bh]
-- @return A 1x2 tensor giving the x,y position of the center
function getCenter(bbox)  local timer=torch.Timer()
  pos = torch.Tensor(1,2):fill(0)
  pos[1][1] = bbox[1][1] + bbox[1][3]/2
  pos[1][2] = bbox[1][2] + bbox[1][4]/2
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return pos
end



--------------------------------------------------------------------------
-- Returns image height and width for a given sequence.
-- @param seqName 	Name of sequence
-- @return height
-- @return width
-- @return nChannels
function getImgRes(seqName)  local timer=torch.Timer()
  imFile = getDataDir() .. seqName .. "/img1/000001.jpg"
  im = image.load(imFile)  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return im:size(2), im:size(3), im:size(1)  -- height x width x nChannel
end

--------------------------------------------------------------------------
--- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
-- @author Andrej Karpathy
-- TODO Modify to accept and profUpdate(debug.getinfo(1,"n").name, timer:time().real) return parameter
function checkCuda() local timer=torch.Timer()
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

  profUpdate(debug.getinfo(1,"n").name, timer:time().real)
end

--------------------------------------------------------------------------
--- Experimental: moves a torch tensor to gpu if needed. Otherwise, converts it to float
function dataToGPU(data) local timer=torch.Timer()
  data=data:float()
  if opt.gpuid >= 0 and opt.opencl == 0 then
    data = data:float():cuda()
  end
  
  if opt.gpuid >= 0 and opt.opencl == 1 then
    data = data:cl()
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data
end

  
  


--------------------------------------------------------------------------
--- Returns number of elements in a table, as well as min. and max. key
-- @return Length of table (number of keys)
-- @return min key
-- @return max key
function tabLen(tab) local timer=torch.Timer()
  local count = 0
  local minKey = 1e5
  local maxKey = -1e5
  for key in pairs(tab) do 
    count = count + 1 
    if key < minKey then minKey = key end
    if key > maxKey then maxKey = key end
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return count, minKey, maxKey
end






--------------------------------------------------------------------------
--- Prints column names for process table
function printTrainingHeadline() local timer=torch.Timer()
  local headline = 
    string.format("%13s%10s%8s%9s%8s%6s","Iter.","Tr. loss","G-norm","tm/btch","l-rate","ETL")
  print(headline)
end

--------------------------------------------------------------------------
--- Prints numbers for current iteration
-- @param sec 	Seconds passed since start
function printTrainingStats(i, me, tl, gn, t, lr, sec) local timer=torch.Timer()
  local secLeft = (sec / (i/opt.max_epochs) - sec)
  local hrsLeft = math.floor(secLeft / 3600)
  local minLeft = torch.round((secLeft % 3600)/60)
  
  print(string.format("%6d/%6d%10.5f%8.2f%8.2fs %.1e%3d:%02d", i, me, tl, gn, t,lr, hrsLeft,minLeft))
end


--------------------------------------------------------------------------
--- Model-specific options in a line
function printModelOptions(opt, modelParams) local timer=torch.Timer()
  local header = ''
  local params = ''
  for k,v in pairs(modelParams) do 
    if string.len(v) > 10 then v = string.sub(v,1,10) end
    header = header..string.format('%11s',v) 
  end
  for k,v in pairs(modelParams) do params = params..string.format('%11d',opt[v]) end
  print(header)
  print(params)
end


--------------------------------------------------------------------------
--- A helper function for evaluation
function matched2d(gt, state, t, map, mID, td) local timer=torch.Timer()
  box1 = gt[t][map]
  box2 = state[t][mID] 
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return boxIoU(box1, box2) >= td
end

--------------------------------------------------------------------------
--- Compute CLEAR MOT Metrics. If one parameter is given, it should be the
-- sequence name. Otherwise two tensors contianing ground truth and results
-- should be provided
-- @param gt	Ground truth or sequence name
-- @param state	State.
function CLEAR_MOT(gt, state, evopt) local timer=torch.Timer()

  if type(gt) == 'string' then
    local seqName = gt
    gt = getGTTracks(seqName)
    state = getResTracks(seqName)
    local imgHeight, imgWidth = getImgRes(seqName)
    gt = gt * imgWidth
  end
  
  -- default opt
  if evopt == nil then
    evopt = {}
    evopt.ev3d = 0
  end
  
  assert(evopt.ev3d==0, "Evaluation in 3D not implemented")
  local matched = matched2d
  local td = 0.5

  local Ngt, Fgt, Dgt = getDataSize(gt)
  local Nst, Fst, Dst = getDataSize(state)
  
  ------------------------------------------------
  -- FLIP DATA, make FxN, as previously
  ------------------------------------------------
  gt = gt:transpose(1,2)
  state = state:transpose(1,2)
  ------------------------------------------------
  
  assert(Fgt == Fst, "Number of frames must be identical")
  assert(Dgt == Dst, "Dimensions do not agree")
  assert(Dgt == 4, "Dimension not equal 4")
  
  local F = Fgt
  
  local M = torch.zeros(F, Ngt)		-- mapping (should be byteTensor)
  local mme = torch.zeros(F)		-- ID switches
  local c   = torch.zeros(F)		-- matches found
  local fp  = torch.zeros(F) 		-- false positives
  local m   = torch.zeros(F)		-- misses, false negatives
  local g   = torch.zeros(F)		
  local d   = torch.zeros(F, Nst)	-- all distances
  local ious= torch.ones(F, Ngt)*2	-- all overlaps
  
  local alltracked = torch.zeros(F, Ngt)
  local allfalsepos = torch.zeros(F, Nst)
  
  local gtInd = gt:narrow(3,1,1):reshape(F, Ngt):ne(0)
  local stInd = state:narrow(3,1,1):reshape(F, Nst):ne(0)  
  
  
  -- main loop through all frames
  for t=1,F do    
    g[t] = torch.sum(gtInd[{{t},{}}],2) -- number of GT in frame
    
    if t>1 then
      -- mappings=find(M(t-1,:));
      local mappings = torch.linspace(1,Ngt, Ngt)[M[{{t-1},{}}]:ne(0)]
--       print(mappings)

      for mm=1,mappings:nElement() do
	local map = mappings[mm]
	if gtInd[t][map]==1 and stInd[t][M[t-1][map]]==1 and matched(gt,state,t,map,M[t-1][map],td) then
	  M[t][map] = M[t-1][map]
	end
	
-- 	print(map)
      end
    end
    -- GTsNotMapped=find(~M(t,:) & gtInd(t,:));
    -- EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));    
    
    local gtInFr=gtInd[{{t},{}}]:eq(1):reshape(Ngt)
    local stInFr=stInd[{{t},{}}]:eq(1):reshape(Nst)
--     local Mapped = M[{{t},{}}]:eq(1)
    local notMapped = M[{{t},{}}]:eq(0)    
    local GTsNotMapped = torch.find(notMapped:cmul(gtInFr):reshape(Ngt))
    local EsNotMapped = torch.Tensor(0)
    if torch.find(stInFr):nElement() > 0 then
      EsNotMapped = torch.setdiff(torch.find(stInFr), M[{{t},{}}])
    end
    
    local allisects = torch.zeros(Ngt,Nst)
    local maxisect = 1e10
    
    if evopt.ev3d > 0 then
      error("oops")
    else
      
      allisects = torch.zeros(Ngt,Nst)
      maxisect = 1e10
--       print(GTsNotMapped)
--       print(EsNotMapped)
      for oo=1,GTsNotMapped:nElement() do
	o = GTsNotMapped[oo]	
	for ee=1,EsNotMapped:nElement() do
	  e = EsNotMapped[ee]
	  allisects[o][e] = boxIoU(gt[t][o], state[t][e])
	end
      end

      local timer = torch.Timer()
      
      -- Hungarian matching
      
-- --     local _, loss = optim.sgd(feval, params, sgd_state)
-- --     local _, loss = optim.cg(feval, params)
      local tmpai = -allisects+1
      tmpai[tmpai:gt(td)] = 1e5
--     local timer = torch.Timer()
--       csvWrite('tmp/cost.txt',tmpai)	-- write cost matrix
--       os.execute("python hung.py")	-- do python munkres
--       ass = csvRead('tmp/ass.txt') 	-- read result
--       ass = torch.Tensor(ass)+1	-- convert back to tensor
--       print(ass)
--     print("a "..timer:time().real)

--     local timer = torch.Timer()
    
--       print(tmpai)
      matstr = ""
      for x=1,tmpai:size(1) do
	for y=1,tmpai:size(2) do
	  matstr = matstr..tmpai[x][y]
	  if y<tmpai:size(2) then matstr=matstr.." " end
	end
	if x<tmpai:size(1) then matstr = matstr.."\n" end
      end
--       print(matstr)

      cmdstr = string.format('echo "%s" | python stdtest.py',matstr)
      local file = assert(io.popen(cmdstr, 'r'))
      local output = file:read('*all')
      file:close()
      
--       print(output) --> Prints the output of the command.

--       ass = csvRead('tmp/ass.txt') 	-- read result
--       ass = torch.Tensor(ass)+1	-- convert back to tensor
      
      pstr = string.gmatch(output, "[%d]+")
      _, n = string.gsub(output, "%(", "a")
--       n=8
      ass = torch.ones(n*2)
      cnt=0

      for k in pstr do
	cnt=cnt+1
	ass[cnt] = tonumber(k)+1
      end
      ass=ass:resize(n,2)

      
--       print(ass)
--     print("b "..timer:time().real)
      
      --       print(t)
--       print(allisects)
--       print(tmpai)
--       print(ass)
--       abort()
    
--       local u = torch.linspace(1,Ngt,Ngt)
--       local v = torch.linspace(1,Ngt,Ngt)
      for mmm=1,ass:size(1) do	
	if allisects[ass[mmm][1]][ass[mmm][2]]>=td then
	  M[t][mmm] = ass[mmm][2]
	end
      end
    end
--     print(M)
--     abort()

    local curtracked = torch.find(M[{{t},{}}]:reshape(Ngt))    
--     print(t)
--     print(curtracked)
--     abort()
    local alltrackers = torch.find(stInFr)        
    local falsepositives = alltrackers
    if curtracked:nElement()>0 then
      local mappedtrackers = torch.intersect(M[t]:index(1,curtracked:long()),alltrackers)    
      falsepositives = torch.setdiff(alltrackers, mappedtrackers)
    end
    
    
    alltracked[{{t},{}}] = M[{{t},{}}]
    for ff=1,falsepositives:nElement() do
      allfalsepos[t][falsepositives[ff]] = falsepositives[ff]
    end
    
    -- mismatch errors
    if t>1 then
      for cc=1,curtracked:nElement() do
	ct = curtracked[cc]
	local lastnotempty = torch.find(M[{{1,t-1},{ct}}]:reshape(t-1))
	if lastnotempty:nElement()>0 then
	  lastnotempty=lastnotempty[-1]
	  if gtInd[t-1][ct] == 1 and M[t][ct] ~= M[lastnotempty][ct] then
	    mme[t] = mme[t] + 1
	  end
	end
      end
    end -- if t>1
    
    c[t] = curtracked:nElement()
    for cc=1,curtracked:nElement() do
      ct = curtracked[cc]
      local eid = M[t][ct]
      if evopt.ev3d == 1 then error("OOps") else
	ious[t][ct]=boxIoU(gt[t][ct], state[t][eid])
      end
    end
  
    fp[t] = torch.sum(stInFr:eq(1)) - c[t]
    m[t] = g[t] - c[t]
    
  end -- for t=1,F  
  
  local missed = torch.sum(m)
  local falsepositives = torch.sum(fp)
  local idswitches = torch.sum(mme)
  
  local MOTP=0
  if evopt.ev3d == 1 then error("OOps") else
--     MOTP = TODO
  end
  local MOTA = (1-(missed+falsepositives+idswitches)/torch.sum(g)) * 100
  local recall = torch.sum(c) / (torch.sum(g)) * 100
  local precision = torch.sum(c) / (falsepositives + torch.sum(c)) * 100 
  local FAR = falsepositives / Fgt
  

  printMetrics(recall,precision,torch.sum(g),falsepositives,missed,idswitches,MOTA,MOTP)
  
--   print(alltracked)
--   print(allfalsepos)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return MOTA
  
--   print(M)
--   print(g)
end

function printMetrics(recall,precision,gbx,falsepositives,missed,idswitches,MOTA,MOTP) local timer=torch.Timer()
  print(string.format("%5s%5s |%5s%5s%5s%5s |%6s%6s","Rcll","Prcn","GTB","FP","FN","ID","MOTA","MOTP"))
  print(string.format("%5.1f%5.1f |%5d%5d%5d%5d |%6.1f%6.1f",
    recall,precision,gbx,falsepositives,missed,idswitches,MOTA,MOTP))
end
--------------------------------------------------------------------------
--- Compute CLEAR MOT Metrics of a sequence
--
function evalSeq(seqName) local timer=torch.Timer()
  gt = getGTTracks(seqName)
  state = getResTracks(seqName)
  local imgHeight, imgWidth = getImgRes(seqName)
  gt = gt * imgWidth
  
  local Ngt,Fgt,Dgt = getDataSize(gt) 
  local Nst,Fst,Dst = getDataSize(state)  
  if Nst == 0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return 0 end
  if Fgt>Fst then
    print(string.format("Padding %d missing frames with zeros",Fgt-Fst))
    state = state:cat(torch.zeros(Nst,Fgt-Fst,Dst),2)
  end  
  
  local timer = torch.Timer()
  local MOTA=CLEAR_MOT(gt,state)
--   print(timer:time().real)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return MOTA
end



-- performance metrics
function accuracy(targets,predictions) local timer=torch.Timer()
   correct=0
   incorrect=0
   predictions=torch.lt(predictions,0.5)
   correct=torch.sum(torch.eq(targets,predictions:type('torch.DoubleTensor')))
   profUpdate(debug.getinfo(1,"n").name, timer:time().real) return correct/targets:size()[1]
end

function auc(targets,pred) local timer=torch.Timer()
   local neg=pred[torch.ne(targets,1)]
   local pos=pred[torch.eq(targets,1)]   
	if neg:nElement() == 0 or pos:nElement() == 0 then
		print('warning, there is only one class')
	end
   local C=0
   for i=1,(#pos)[1] do
      for j=1,(#neg)[1] do
         if neg[j]<pos[i] then
            C=C+1
         elseif neg[j]==pos[i] then
            C=C+0.5
         end
      end
   end
   local AUC=C/((#neg)[1]*(#pos)[1])
   profUpdate(debug.getinfo(1,"n").name, timer:time().real) return AUC
end

--------------------------------------------------------------------------
--- Pads a tensor with zeros along a certain dimension if necessary
-- @param data	The tensor.
-- @param N	The desired dimensionality
-- @param dim 	The dimension along which to pad
function padTensor(data, N, dim) local timer=torch.Timer()
  assert(data:nDimension()>=3, "At least 3-dim tensor expected")
  local R,C,D = getDataSize(data)
  
  if N<=data:size(dim) then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data end 	-- nothing to do
  
  if dim == 1 then
    profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data:cat(torch.zeros(N-R,C,D),1)
  elseif dim == 2 then
    profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data:cat(torch.zeros(R,N-C,D),2)
  elseif dim == 3 then
    profUpdate(debug.getinfo(1,"n").name, timer:time().real) return data:cat(torch.zeros(R,C,N-D),3)
  end
  
end


--------------------------------------------------------------------------
--- Save checkpoint (convert to CPU first)
function saveCheckpoint(savefile, tracks, detections, protos, opt, trainLosses, time, it) local timer=torch.Timer()  
  savefile = savefile or string.format('bin/model.t7')
  print('saving model to ' .. savefile)
  local checkpoint = {}
  checkpoint.detections = detections:float()
  checkpoint.gt = tracks:float()
  for k,v in pairs(protos) do protos[k] = protos[k]:float() end
  checkpoint.protos = protos
  checkpoint.opt = opt
  checkpoint.trainLosses = trainLosses
  checkpoint.i = opt.max_epochs
  checkpoint.epoch = opt.max_epochs  
  checkpoint.time = time
  checkpoint.it = it
  torch.save(savefile, checkpoint)
end


function hungarian(mat) local timer=torch.Timer()
  local t2=torch.Timer()
  matstr = ""
  for x=1,mat:size(1) do
    for y=1,mat:size(2) do
      matstr = matstr..mat[x][y]
      if y<mat:size(2) then matstr=matstr.." " end
    end
    if x<mat:size(1) then matstr = matstr.."\n" end
  end
  profUpdate("Hun. matrix", t2:time().real)
  
  local t2=torch.Timer()
  cmdstr = string.format('echo "%s" | python stdtest.py',matstr)
  profUpdate("Python hun", t2:time().real)
  
  local t2=torch.Timer()
  local file = assert(io.popen(cmdstr, 'r'))
  profUpdate("Pipe open", t2:time().real)
  
  local t2=torch.Timer()
  local output = file:read('*all')
  file:close()
  profUpdate("Pipe read", t2:time().real)
  
  
  
    
  local t2=torch.Timer()
  pstr = string.gmatch(output, "[%d]+")
  _, n = string.gsub(output, "%(", "a")
--       n=8
  ass = torch.ones(n*2)
  cnt=0
  

  for k in pstr do
    cnt=cnt+1
    ass[cnt] = tonumber(k)+1
  end
  profUpdate("Parsing python", t2:time().real)
  
  ass=ass:resize(n,2)
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return ass
end      


--------------------------------------------------------------------------
--- Returns ground truth and detections for a set of sequences
-- @param seqTable	The table of sequences to be read
-- @param maxTargets	Max number of targets (per frame) to pad tensors
-- @param maxDets	Max number of detections (per frame) to pad tensors
-- @param cropFrames	Whether to trim the sequence to a time window
function getTracksAndDetsTables(seqTable, maxTargets, maxDets, cropFrames) local timer=torch.Timer()
  cropFrames = cropFrames or true
  
  local allTracksTab = {}
  local allDetsTab = {}  
  for _, seqName in pairs(seqTable) do
    local tracks = getGTTracks(seqName)
    local detections = getDetTensor(seqName)

    if stateDim ~= nil then 
      tracks=tracks:sub(1,-1,1,-1,1,stateDim)
      detections=detections:sub(1,-1,1,-1,1,stateDim)
    end
    
    local ts = 1 -- first frame
    if opt ~= nil and cropFrames then
      local F = opt.temp_win -- subseq length
      tracks = selectFrames(tracks,ts,ts+F-1)
      detections = selectFrames(detections,ts+1,ts+F)
    end

    -- pad tensors with zeros if necessary
    tracks = padTensor(tracks, maxTargets, 1)
    detections = padTensor(detections, maxDets, 1)
    
    -- trim tracks to maxTargets if necessary
    if tracks:size(1) > maxTargets then tracks = tracks:narrow(1,1,maxTargets) end
    
    if detections:size(1) > maxDets then detections = detections:narrow(1,1,maxDets) end
    
    table.insert(allTracksTab, tracks)
    table.insert(allDetsTab, detections)
  end
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return allTracksTab, allDetsTab
end


--------------------------------------------------------------------------
--- Returns pid of X Server if available, nil otherwise
function xAvailable() local timer=torch.Timer()
  local f = io.popen('pidof X')
  local t = f:read()
  f:close()
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return t
end


--------------------------------------------------------------------------
--- Get a smoothed version of the training loss for plotting
-- @param i		iteration
-- @param plotLossFreq	plotting frequency
-- @param trainLosses	A table containing (iteration number, training loss) pairs
function getLossPlot(i, plotLossFreq, trainLosses) local timer=torch.Timer()
  
  local addOne = 0-- plus one for iteration=1
  local plot_loss = torch.Tensor(math.floor(i / (plotLossFreq))+addOne) 
  local plot_loss_x = torch.Tensor(math.floor(i / (plotLossFreq))+addOne)
  if i==1 then plot_loss, plot_loss_x = torch.ones(1), torch.ones(1) end

  local cnt = 0
  local tmp_loss = 0
  local globCnt = 0
  for a,b in pairs(trainLosses) do
    cnt = cnt+1
    tmp_loss = tmp_loss + b
    -- compute avaerage and reset
--     if a==1 or (a % plotLossFreq) == 0 then
    if (a % plotLossFreq) == 0 then            
      globCnt = globCnt + 1
      tmp_loss = tmp_loss / cnt
      plot_loss[globCnt] = tmp_loss
      plot_loss_x[globCnt] = a
      cnt = 0
      tmp_loss = 0
    end
  end      
--   plot_loss_x, plot_loss = getValLossPlot(trainLosses)
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return plot_loss_x, plot_loss
  
end

--------------------------------------------------------------------------
--- Get a smoothed version of the training loss for plotting
function getValLossPlot(val_losses) local timer=torch.Timer()
  local tL = tabLen(val_losses)
  local plot_val_loss = torch.Tensor(tL)
  local plot_val_loss_x = torch.Tensor(tL)

  local orderedKeys = {}
  for k in pairs(val_losses) do table.insert(orderedKeys, k) end
  table.sort(orderedKeys)

  for cnt=1,tL do	
    plot_val_loss[cnt]=val_losses[orderedKeys[cnt]]
    plot_val_loss_x[cnt]=orderedKeys[cnt]
  end  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return plot_val_loss_x, plot_val_loss
end


--------------------------------------------------------------------------
--- Prints depending on the debug level set in opt.verbose
-- @param message 	The message to be printed
-- @param vl		The verbosity level (0=none, 1=warning, 2=info, 3=debug)
function pm(message, vl) local timer=torch.Timer()
  vl=vl or 2
  if vl <= opt.verbose then
    print(message)
  end
end

--------------------------------------------------------------------------
--- Prints all options (TODO Needs prettyfying)
function printOptions(opt) local timer=torch.Timer()
  local modelOpt = {
    'rnn_size','num_layers'
  }
  local hide = {
    ['gpuid']=0,['verbose']=0,['savefile']=0,['checkpoint_dir']=0,
    ['print_every']=0,['plot_every']=0,['eval_val_every']=0,['profiler']=0
  }
  
  for k,v in pairs(opt) do       
    if hide[k] == nil then
      pm(string.format("%21s  %s",k,tostring(v)),2)
    end
  end
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
function getCheckptFilename(base, opt, modelParams) local timer=torch.Timer()
  local tL = tabLen(modelParams)	-- how many relevant parameters
  
  local ext = '.t7'			-- file extension
  local dir = 'bin/'			-- directory  
  local signature = ''
  
  for i=1,tL do
    local p = opt[modelParams[i]]
    local pr = ''			-- prepend suffix
    if modelParams[i] == 'rnn_size' 		then pr='r' 
    elseif modelParams[i] == 'num_layers' 	then pr='l' 
    elseif modelParams[i] == 'max_n' 		then pr='n' 
    elseif modelParams[i] == 'state_dim' 	then pr='d' 
    elseif modelParams[i] == 'loss_type' 	then pr='lt'
    elseif modelParams[i] == 'batch_size' 	then pr='b' end
    signature = signature .. pr
    if p==torch.round(p) then
      signature = signature .. string.format('%d',p)	-- append parameter
    else 
      signature = signature .. string.format('%.2f',p)	-- append parameter
    end
    if i<tL then signature = signature .. '_' end	-- append separator 
  end
  
  fn = dir .. base .. '_' .. signature .. ext			-- append extension
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return fn, dir, base, signature, ext
end



-----------------------------------
-- Synthetic Trajectory Sampling --
-- NOTE This is rather ad hoc. Needs quite a bit of work
-- NOTE Only 1-D for now
function getOneBatch() local timer=torch.Timer()
  assert(opt.state_dim == 1, "Trajectory sampling not implemented for dim ~= 1")
    
  local tr = torch.zeros(opt.max_n, opt.temp_win, opt.state_dim)
  local randF1=0.005
  local randF2=0.005
  local maxV = 0.025
  
  
  
  -- uniform
  local trackLocs = torch.linspace(0.1,0.9,opt.max_n)
  
  -- random 
  local trackLocs = torch.rand(opt.max_n)
  
  local fillTracks = torch.random(opt.max_n) -- generate n (1 <= n <= N) tracks
--   local fillTracks = maxTargets -- generate n = N tracks
  -- fill in tracks
  for i=1,fillTracks do
    local trackLoc = trackLocs[i]
    local s1 = trackLoc+randF1*torch.randn(1):squeeze() -- start
    local totV = maxV * math.random()*opt.temp_win 
    if math.random()<0.5 then totV = -totV end
    local e1 = s1 + totV + randF2 * torch.randn(1):squeeze()
    
    if s1>e1 then s1,e1 = e1,s1 end
    
    tr[{{i},{},{}}] = torch.linspace(s1,e1,opt.temp_win)
    
    -- add slight curve
    if math.random()<0.95 and opt.temp_win > 5 then
      local knick = torch.random(opt.temp_win-5)+3
      s1=tr[{{i},{knick},{1}}]:squeeze()
      local totV = maxV * math.random()*(opt.temp_win - knick )
      e1 = s1 + 2*totV + randF2 * torch.randn(1):squeeze()
      if s1>e1 then s1,e1 = e1,s1 end
--       print(knick)
--       print(tr[{{i},{knick,opt.temp_win},{}}])
--       print(torch.linspace(s1,e1,opt.temp_win-knick))
      tr[{{i},{knick,opt.temp_win},{}}] = torch.linspace(s1,e1,opt.temp_win-knick+1)
    end
  end
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tr
end


----------------------------------------
-- Learn generative trajectory  model --
function learnTrajModel(trTable) local timer=torch.Timer()
  local tL = tabLen(trTable)
  _,_,D = getDataSize(trTable[1])
  
  -- number of features for each trajectory
  -- 2 start / end frame
  -- 2 * D start / end state
  -- D mean velocity
  -- 
  
  local nFeat = 2 + 2*D + D
--   local nFeat = 2 + 2*D + 0
  local m = torch.zeros(1,nFeat) -- dummy row, rest is catted on bottom
--   print(m)
--   abort()
  
  
  for s=1,tL do
    local tracks = trTable[s]
    
    N,F,D = getDataSize(tracks)
    
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
  end
  
--   print(m)
  m = m:sub(2,-1) -- remove first dummy row
--   print(m)
  mmean = torch.mean(m,1)
  mstd = torch.std(m,1)
  if tL<=1 then mstd = torch.mul(mmean, 1) end -- avoid NaNs if one track only
--   print(m)
--   print(mmean)
--   print(mstd)
--   sleep(1)
--   print(torch.mean(m,1))
--   print(torch.std(m,1))
  
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return torch.cat(mmean, mstd, 1)
  
end

--- Return a sample from a 1D normal distribution
function sampleNormal(mean, std) local timer=torch.Timer()
--   print(torch.randn(1):squeeze())
  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return std*torch.randn(1):squeeze() + mean
end


function sampleTrajectory(m, stateDim) local timer=torch.Timer()
  
  local validTraj = false
  
--   while not validTraj do
      -- create zero track of length and dim
    trZ = torch.zeros(1, opt.temp_win, stateDim)

    
    -- sample start / end frame
    local featInd = 1
  --   print()
    local s = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    s=1
--     if s<1 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return trZ end
    
    featInd = 2;
    local e = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    e=opt.temp_win
    if e>opt.temp_win then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return trZ end    
    local F = e-s+1
    if F<=0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) return trZ end
    
    tr = torch.zeros(opt.temp_win, stateDim)
    
    for d = 1,stateDim do
      -- get linear sample
      featInd = 3+2*(d-1)
  --     print(d, m[1][featInd])
      local sx = sampleNormal(m[1][featInd], m[2][featInd])
      
      featInd = 3 + 2*stateDim + (d-1)
  --     print(featInd)
  --     print(m)
      local mv = sampleNormal(m[1][featInd], m[2][featInd])
  --     print(mv)
      local ex = sx + mv * F
  --     if sx>ex then sx,ex = ex,sx end -- this might be a bug! WARNING
      for t=s,e do tr[t][d] = sx + mv * (t-s) 
  --   print(tr[t][d])
    end
  --     tr[{{s,e},{d}}] = torch.linspace(sx,ex,F)

  --     featInd = 4+2*(d-1)
  --     local ex = sampleNormal(m[1][featInd], m[2][featInd])
  --     if sx>ex then sx,ex = ex,sx end
  --     tr[{{s,e},{d}}] = torch.linspace(sx,ex,F)
      
      
    end
--   end

  profUpdate(debug.getinfo(1,"n").name, timer:time().real) return tr:reshape(1,opt.temp_win, stateDim)
  
  
end


function profUpdate(fname, ftime) local timer=torch.Timer()
  if profTable[fname] == nil then
    profTable[fname] = torch.Tensor({0,0})
  end
  profTable[fname] = profTable[fname] + torch.Tensor({ftime, 1})
end