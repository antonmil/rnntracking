--------------------------------------------------------------------------
--- Orders detections in one frame according to their distance to tracks.
-- -- -- function orderDets(detections, tracks, daMode)
  -- -- -- daMode = daMode or 2 -- 1=greedy, 2=hungarian
-- -- -- --   daMode = 2
  
  -- -- -- local Ndet,Fdet = getDataSize(detections)
  -- -- -- local N,F = getDataSize(tracks)
  -- -- -- if detections:nDimension() > 2 then
    -- -- -- detections = detections:narrow(3,1,1):reshape(Ndet,Fdet) -- use X dim only
  -- -- -- end
  -- -- -- if tracks:nDimension() > 2 then tracks = tracks:narrow(3,1,1):reshape(N,F) end

  -- -- -- local detOrdered = detections:clone()
  -- -- -- local ndet = detections:size(1)
  -- -- -- local da = torch.zeros(ndet)
  
  -- -- -- -- GREEDY DATA ASSOCIATION
  -- -- -- if daMode == 1 then
    -- -- -- for t=1,opt.temp_win do
-- -- -- --     for t=1,1 do
      -- -- -- for i = 1,tracks:size(1) do
	-- -- -- -- dist of track i to all det
	-- -- -- cloneTrack = torch.repeatTensor(tracks[{{i},{t}}],ndet,1)
	-- -- -- distVec = torch.abs(cloneTrack-detections[{{},{t}}])
	-- -- -- sDist,cDet = torch.min(distVec,1) -- find closest detection to track i
	-- -- -- sDist=sDist[1][1]; cDet=cDet[1][1]
	-- -- -- detOrdered[i][t] = detections[cDet][t]
	-- -- -- da[cDet]=i
	
      -- -- -- end
    -- -- -- --   print("Smallest dist: "..sDist..". Nearest Det: "..cDet)    
    -- -- -- end
  -- -- -- end
  
  -- -- -- if daMode == 2 then
    -- -- -- t=1
    -- -- -- -- create cost matrix
    -- -- -- cost = torch.Tensor(tracks:size(1),ndet)
    -- -- -- for i=1,tracks:size(1) do
	-- -- -- cloneTrack = torch.repeatTensor(tracks[{{i},{t}}],ndet,1)
	-- -- -- distVec = torch.abs(cloneTrack-detections[{{},{t}}])
	-- -- -- cost[i] = distVec:t()
    -- -- -- end
    -- -- -- -- we need 1 / dist as cost? NO
  -- -- -- --   cost = torch.cdiv(torch.ones(cost:size(1), cost:size(2)), cost)
    
    -- -- -- cost[cost:gt(0.03)]=10		-- jump to max if over threshold
    -- -- -- cost[cost:gt(10)]=10		-- avoid Inf
  -- -- -- --   print(cost)
    
-- -- -- --     csvWrite('tmp/cost.txt',cost:t())	-- write cost matrix
-- -- -- --     os.execute("python hung.py")	-- do python munkres
-- -- -- --     ass = csvRead('tmp/ass.txt') 	-- read result
-- -- -- --     ass = torch.Tensor(ass)+1	-- convert back to tensor
-- -- -- --     print('---------------------')
-- -- -- --     print(detections)
-- -- -- --     print(tracks)
-- -- -- --     print(cost)
-- -- -- --     print(ass)
    -- -- -- ass = hungarian(cost)
-- -- -- --     print(ass)
    
    -- -- -- for i=1,ndet do
      -- -- -- detOrdered[i] = detections[ass[i][2]][t]
      -- -- -- da[i]=ass[i][2]
    -- -- -- end
  -- -- -- --   print(detOrdered)
  -- -- -- --   sleep(1)
  -- -- -- --   print(ass)
  -- -- -- --   abort()
    
  -- -- -- end -- daModa
  
  -- -- -- return detOrdered, da
-- -- -- end

--------------------------------------------------------------------------
function getTrack(gt,qid)
  
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
  
  return trackT, ts, te
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
  
  local gt = getGT(seqName)	-- get raw GT table
  
  
  local imgHeight, imgWidth = getImgRes(seqName) -- required for normalization
  imgWidth = 1
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
  return tracks
end

-- special ID for false alarm
function getFALabel()
  return -1
end

--------------------------------------------------------------------------
--- Naively associate detections.
--  For each detection find closest track and if it is below a threshold, take it.
--
-- @param detections	FdxNd tensor: current detections in arbitrary order
-- @param tracks	FxN tensor:   tracks to associate detections to
-- @param thr 		threshold to consider inlier
-- @return 		FdxNd int tensor with det-to-track IDs
function associateDetections(detections, tracks, thr)
--   detections = selectStateDim(detections,1) 	-- select x as primary dim
--   tracks = selectStateDim(tracks,1)
  
  local Ndet,Fdet, stateDim = getDataSize(detections)	-- detection tensor size
  local N,F = getDataSize(tracks)		-- tracks tensor size
  local da = torch.Tensor(Ndet,Fdet):fill(getFALabel())	-- new tensor w/ assc. detIDs
  da = dataToGPU(da)
  da[torch.ne(selectStateDim(detections,1),0)] = getFALabel()	-- fill in all as false alarm
  
  thr = thr or 0.3*stateDim  
  
--   if stateDim == 4 then thr = 0.5 end
  
  -- loop through all detections
  for t=1,math.min(Fdet,F) do
    for d=1,Ndet do      
      -- find closest track to this particular detection
      if detections[d][t][1] ~= 0 then -- ignore dummy detections
	local sDist, cTr = findClosestTrack(detections[d][t], tracks[{{},{t}}])
-- 	print(detections[d][t])
-- 	print(tracks[{{},{t}}])
-- 	print(t,d,sDist,cTr,thr)
	if sDist < thr then	-- associate if below threshold      
	  da[d][t] = cTr
-- 	  print(da)
	end
      end
    end
  end  
  
--   print(da)
  return da
  
end

--------------------------------------------------------------------------
--- Compute distance between two states
-- The actual computation depends on the state dimension.
-- 1D and 2D states invoke Euclidean distance
-- 4D state assumes two bounding boxes and computes the IoU score
-- @param state1
-- @param state2
function computeDistance(state1, state2)
  error('computeDistance not implemented')  
end

--------------------------------------------------------------------------
--- Pad tracks or detections to a consistent size
-- @param tracks	The tracks tensor
-- @param detections	The detections tensor
-- @param da		The det-to-ID association tensor
-- @param maxTracks	Option: The max number of tracks / classes to pad to
function unifyData(tracks, detections, da, maxTracks)
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
	if tracks[i][t][1]~=0 then da[i][t] = getFALabel() end
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
	if tracks[i][t][1]~=0 then da[i][t] = getFALabel() end
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
  
  return tracks, detections, da
  
end

--------------------------------------------------------------------------
--- A helper function for evaluation
function matched2d(gt, state, t, map, mID, td)
  box1 = gt[t][map]
  box2 = state[t][mID] 
  
  return boxIoU(box1, box2) >= td
end

--------------------------------------------------------------------------
--- Compute CLEAR MOT Metrics. If one parameter is given, it should be the
-- sequence name. Otherwise two tensors contianing ground truth and results
-- should be provided
-- @param gt	Ground truth or sequence name
-- @param state	State.
function CLEAR_MOT(gt, state, evopt)

  if type(gt) == 'string' then
    local seqName = gt
    gt = getGTTracks(seqName)
    state = getResTracks(seqName)
    local imgHeight, imgWidth = getImgRes(seqName)
    gt = gt * imgWidth
--     print(gt
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
      local maxCost = 1e5
      tmpai[tmpai:gt(td)] = maxCost
--     local timer = torch.Timer()
--       csvWrite('tmp/cost.txt',tmpai)	-- write cost matrix
--       os.execute("python hung.py")	-- do python munkres
--       ass = csvRead('tmp/ass.txt') 	-- read result
--       ass = torch.Tensor(ass)+1	-- convert back to tensor
--       print(ass)
--     print("a "..timer:time().real)

--     local timer = torch.Timer()
    
--       print(tmpai)
-- -- --       matstr = ""
-- -- --       for x=1,tmpai:size(1) do
-- -- -- 	for y=1,tmpai:size(2) do
-- -- -- 	  matstr = matstr..tmpai[x][y]
-- -- -- 	  if y<tmpai:size(2) then matstr=matstr.." " end
-- -- -- 	end
-- -- -- 	if x<tmpai:size(1) then matstr = matstr.."\n" end
-- -- --       end
-- -- -- --       print(matstr)
-- -- -- 
-- -- --       cmdstr = string.format('echo "%s" | python stdtest.py',matstr)
-- -- --       local file = assert(io.popen(cmdstr, 'r'))
-- -- --       local output = file:read('*all')
-- -- --       file:close()
-- -- --       
-- -- -- --       print(output) --> Prints the output of the command.
-- -- -- --       ass = csvRead('tmp/ass.txt') 	-- read result
-- -- -- --       ass = torch.Tensor(ass)+1	-- convert back to tensor
-- -- --       
-- -- --       pstr = string.gmatch(output, "[%d]+")
-- -- --       _, n = string.gsub(output, "%(", "a")
-- -- -- --       n=8
-- -- --       ass = torch.ones(n*2)
-- -- --       cnt=0
-- -- -- 
-- -- --       for k in pstr do
-- -- -- 	cnt=cnt+1
-- -- -- 	ass[cnt] = tonumber(k)+1
-- -- --       end
-- -- --       ass=ass:resize(n,2)
--       print(tmpai)
      -- padding
--       local R,C = getDataSize(tmpai)
--       if R > C then -- more rows than columns
-- 	tmpai = tmpai:cat(torch.ones(R,R-C)*0,2) -- pad cols
--       elseif C > R then -- more columns than rows
-- 	tmpai = tmpai:cat(torch.ones(C-R,C)*0,1) -- pad rows
--       end
--       print(tmpai)
      local ass = hungarianL(tmpai)
      
--       abort('CLEAR')
      
      -- make it work with HungarianL
-- 	local ass = hungarianP(tmpai)
	
      
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
-- 	print(allisects:size())
-- 	print(ass[mmm][1])
-- 	print(ass[mmm][2])
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
  

  
  
  local metTensor = torch.Tensor({recall,precision,torch.sum(g),falsepositives,missed,idswitches,MOTA,MOTP})
  printMetrics(metTensor)
  
--   print(alltracked)
--   print(allfalsepos)
  
--   return MOTA
  return metTensor
  
--   print(M)
--   print(g)
end

--------------------------------------------------------------------------
--- TODO docs
function getEmptyMetrics(nGTBoxes)
  return torch.Tensor({0,0,nGTBoxes,0,nGTBoxes,0,0,0})
end

--------------------------------------------------------------------------
--- TODO docs
function printMetricsHeader()
  print(string.format("%5s%5s |%5s%5s%5s%5s |%6s%6s","Rcll","Prcn","GTB","FP","FN","ID","MOTA","MOTP"))
end

--------------------------------------------------------------------------
--- TODO docs
function printMetrics(mets, ph)
  -- recall,precision,gbx,falsepositives,missed,idswitches,MOTA,MOTP
  ph = ph or 1 -- print header
  if ph>0 then printMetricsHeader() end
  
  mets=mets:view(8)
--   print(mets)
  print(string.format("%5.1f%5.1f |%5d%5d%5d%5d |%6.1f%6.1f",
    mets[1],mets[2],mets[3],mets[4],mets[5],mets[6],mets[7],mets[8]))
end

--------------------------------------------------------------------------
--- Compute CLEAR MOT Metrics of a sequence
--
function evalSeq(seqName, resDir, cropGT)
  resDir = resDir or getResDir()
  local cropGT = cropGT or 0
  
  gt = getGTTracks(seqName)
  state = getResTracks(seqName, resDir)
  local imgHeight, imgWidth = getImgRes(seqName)
--   gt = gt * imgWidth
--   print(gt[{{1},{1,10},{}}])
--   print(state[{{3},{1,8},{}}])
--   abort()
  
  local Ngt,Fgt,Dgt = getDataSize(gt) 
  local Nst,Fst,Dst = getDataSize(state)  
  local nGTBoxes = torch.sum(torch.ne(gt:narrow(3,1,1):squeeze(),0))
  if Nst == 0 then 
    local emptyMet = getEmptyMetrics(nGTBoxes)
    printMetrics(emptyMet)
    return emptyMet
  end
  if Fgt>Fst and cropGT ~= 0 then
    print(string.format("cropping GT"))
    gt = selectFrames(gt, 1, Fst)
  elseif Fgt>Fst and cropGT == 0 then
    print(string.format("Padding %d missing frames with zeros",Fgt-Fst))
    state = state:cat(torch.zeros(Nst,Fgt-Fst,Dst),2)
  else
    error('Result has more frames than GT?!?!')
  end  
  
  local timer = torch.Timer()
  local mets=CLEAR_MOT(gt,state)
--   print(timer:time().real)
  return mets
end

-- performance metrics
function accuracy(targets,predictions)
   correct=0
   incorrect=0
   predictions=torch.lt(predictions,0.5)
   correct=torch.sum(torch.eq(targets,predictions:type('torch.DoubleTensor')))
   return correct/targets:size()[1]
end

function auc(targets,pred)
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
   return AUC
end

function hungarianP(mat)
  matstr = ""
  for x=1,mat:size(1) do
    for y=1,mat:size(2) do
      matstr = matstr..mat[x][y]
      if y<mat:size(2) then matstr=matstr.." " end
    end
    if x<mat:size(1) then matstr = matstr.."\n" end
  end

  cmdstr = string.format('echo "%s" | python stdtest.py',matstr)
  local file = assert(io.popen(cmdstr, 'r'))
  local output = file:read('*all')
  file:close()
  
  
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
  return ass
end      


-----------------------------------
-- Synthetic Trajectory Sampling --
-- NOTE This is rather ad hoc. Needs quite a bit of work
-- NOTE Only 1-D for now
function getOneBatch()
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
  
  return tr
end



-- TODO to merge with normalizeData
function normalizeDataSingle(data, det, backwards)
  error('still used?')
  backwards = backwards or false
  
  dataRet = data:clone()
  local trueDet = selectStateDim(det,1):ne(0)	-- existing detections mask
  local trueTar = selectStateDim(data,1):ne(0)		-- existing target mask
  local dMean, dStd = torch.zeros(stateDim), torch.zeros(stateDim)
  if torch.sum(trueDet) > 1 then 			-- handle special all-zeros (no dets) case
    dMean = torch.zeros(stateDim)			-- mean vector per dimension
    dStd = torch.ones(stateDim)			-- std vector
    
    for d=1,stateDim do
      local dimDet = selectStateDim(det,d)[trueDet]    

      local divFactor = torch.std(dimDet)
      local shiftFactor = torch.mean(dimDet)
--       print(opt.norm_std)
--       print(opt.norm_mean)
--       abort()
      if opt.norm_std ~= nil and opt.norm_std > 0 then divFactor = opt.norm_std end
      if opt.norm_mean ~= nil and opt.norm_mean == 0 then shiftFactor = 0 end
	
      dStd[d] = divFactor
      dMean[d] = shiftFactor


--       dStd[d] = 1
--       dMean[d] = 0
--      dStd[d] = 100
--       print(torch.sum(trueDet))
--       print(dimDet)
      assert(dStd[d]>0, string.format('std. dev. not positive: %f', dStd[d]))
    end
--     print(dMean)
--     print(dStd)
    for d=1,stateDim do
      if backwards then
	dataRet[{{},{},{d}}] = dataRet[{{},{},{d}}] * dStd[d] + dMean[d]
      else
	dataRet[{{},{},{d}}] = (dataRet[{{},{},{d}}] - dMean[d]) / dStd[d]            
      end
    end
    
    local N,F,D = getDataSize(data)
--     for i=1,N do
--       for t=1,F do
-- 	if trueTar[i][t]==0 then dataRet[i][t] = 0 end
--       end
--     end
    newTT = trueTar:reshape(N,F,1):expand(N,F,D)
    dataRet[newTT:eq(0)]=0
    
  end
  

  return dataRet, dMean, dStd
end


--------------------------------------------------------------------------
--- Runs model on all training sequences of the 2DMOT15 Challenge
function runMOT15LUA(runScript, modelName, modelSign)
  
--   local runScript = 'rnnTrackerBTC_det.lua'

  local sequences = {
    'TUD-Stadtmitte',
    'TUD-Campus',
    'PETS09-S2L1',
    'ETH-Bahnhof',
    'ETH-Sunnyday',
    'ETH-Pedcross2',
    'ADL-Rundle-6',
    'ADL-Rundle-8',
    'KITTI-13',
    'KITTI-17',
    'Venice-2',
  }
  
  local resDir = getResDir(modelName, modelSign)

  for _,seqName in pairs(sequences) do
    
    cmdstr = string.format('th %s -model_name %s -model_sign %s -seq_name %s -suppress_x 1 -length 0 -det_fail 0.2 -gtdet 0 -clean_res 1 -eval 0 -crop_gt 0', 
      runScript, modelName, modelSign, seqName)

  --   print(cmdstr)
    print('Running RNN on '..seqName)
    os.execute(cmdstr)
  end

  local nMets = 8
  local allmets = torch.Tensor(1,nMets)
--   sopt = opt
--   print(opt)
--   print(sopt)
  for _,seqName in pairs(sequences) do
    print('\nEvaluating '..seqName)
    local mets = evalSeq(seqName, resDir, opt.crop_gt):view(1,-1)
    allmets = allmets:cat(mets,1)			-- attach on bottom
  end
  allmets = allmets:sub(2,-1) 			-- remove first row

  print('\nAll results')
  printMetricsHeader()
  for k,seqName in pairs(sequences) do
    printMetrics(allmets:sub(k,k), 0)
  end
  print('-----------------------------------------------')

  -- print(allmets)
  local meanmets = torch.mean(allmets,1)
  meanmets[{{1},{3,6}}] = torch.round(meanmets[{{1},{3,6}}]) -- round integer metrics
  printMetrics(meanmets, 0)
  return meanmets
end




-- TODO docs
function synthesizeData(trackModels, nSynth, mbSize)
  local trTracksTab, trDetsTab, trOrigDetsTab = {}, {}, {}
  
  local tt = false
  if opt.trim_tracks ~= nil and opt.trim_tracks > 0 then tt = true end
--   print(trackModels, nSynth, mbSize)
--   abort()
  local tL = tabLen(trackModels) 	-- how many modes?
--   print(tL)
--   abort()
  for n=1,nSynth do
    
    local alltr = torch.zeros(1, opt.temp_win, opt.state_dim)
    local alldet = torch.zeros(1, opt.temp_win, opt.state_dim)
    local allodet = torch.zeros(1, opt.temp_win, opt.state_dim)
    for m=1,mbSize do
      local trMode = math.random(tL)
      local trackModel = trackModels[trMode]	-- pick one model
      local tr = torch.zeros(1,opt.temp_win, opt.state_dim)

      local ntracks = torch.random(opt.max_n)
      if opt.fixed_n ~= 0 then ntracks = opt.max_n end
--       print(ntracks)
--       abort()
      for ss = 1,ntracks do -- at least one
-- 	print(n,m,ss)

	local sampleTrack = sampleTrajectory(trackModel, opt.state_dim)
	local trajS = math.random(opt.temp_win/4)
	local trajE = opt.temp_win - math.random(opt.temp_win/4)+1
	
-- 	print(trajS, trajE)
-- 	sleep(1)
-- 	abort()
	----------
	-- trim tracks
	if tt then
	  sampleTrack[{{1},{1,trajS},{}}]=0 -- WARNING EXPERIMENT
	  if trajE-trajS>1 and trajE<opt.temp_win then
	    sampleTrack[{{1},{trajE,opt.temp_win},{}}]=0 -- WARNING EXPERIMENT
	  end
	end
	----------
-- 	print(sampleTrack)
-- 	abort()
    --     print(trackModel)
	tr = tr:cat(sampleTrack, 1)
      end  
      tr=tr:sub(2,-1) -- remove first dummy row
      tr = padTensor(tr, opt.max_n, 1)
      if tr:size(1) > opt.max_n then tr = tr:narrow(1,1,opt.max_n) end  
      
      local det = tr:clone()
      det = padTensor(det, opt.max_m, 1)
--       det=perturbDetections(det, opt.det_noise, opt.det_fail, opt.det_false) 
      allodet = allodet:cat(det, 1)
      
      
      if opt.reshuffle_dets > 0 then  det = reshuffleDets(det) end
      
      alltr = alltr:cat(tr, 1)
      alldet = alldet:cat(det, 1)
      
      
    end
    alltr = alltr:sub(2,-1)
    alldet = alldet:sub(2,-1)
    allodet = allodet:sub(2,-1)

    table.insert(trOrigDetsTab, allodet)
    table.insert(trTracksTab, alltr)
    table.insert(trDetsTab, alldet)
    

    
  end
  return trTracksTab, trDetsTab, trOrigDetsTab
end


function getSnappedPos(sampledTraj, detections)
  --   local otracks = orderTracks(sampledTraj, predLab)
  local otracks = sampledTraj:clone()
  local snappedTracks = otracks:clone():fill(0)
  local N,F,D = getDataSize(otracks)
  local thr = 0.03*stateDim
  for t=1,F do
--     print(t)
    local dets = detections[{{},{t+1}}] -- dets of current minibatch
    local distmat=torch.zeros(maxDets, maxTargets)

    for det=1,maxDets do
      if torch.sum(torch.abs(dets[det][1]))==0 then
	distmat[det] = sopt.dummy_weight
      else      
	for pred=1,maxTargets do
	  for dd=1,math.min(2,stateDim) do
	    local absd = math.abs(otracks[pred][t][dd] - dets[det][1][dd])
	    distmat[det][pred] = distmat[det][pred] + absd*absd		  
	  end
	end
      end
    end

    ass = hungarianL(distmat)

    trOrig = otracks:clone()
--     otracks:fill(0)
    for i=1,ass:size(1) do 
      snappedTracks[{{ass[i][2]},{t},{}}] = dets[{{ass[i][1]},{1},{}}] 
    end            
  end

  return snappedTracks
end