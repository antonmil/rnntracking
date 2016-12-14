--[[
This file contains functions for data handling.
Loading, generating, preparing train and test data
--]]

-------------------------------------------------------------------------
--- load / generate and pre-process data
function prepareData(mode, sequences, trainingSequences, singleBatch)
  assert(mode=='train' or mode=='validation' or mode=='real', 'unknown mode')

  pm('Preparing '..mode..' data...');

  local nSamples = opt.synth_training
  if mode=='validation' then nSamples = opt.synth_valid end


  -- load?
  if LOAD and not TESTING and (mode=='train' or mode=='validation') and loadSynthTraining(opt,mode) then
    local dataFileName =  getTrainingDataFilename(opt, dataParams, mode)
    print('Loading data file '..dataFileName)
    local trData = torch.load(dataFileName)
    local TracksTab, DetsTab, LabTab, ExTab, DetExTab, SeqNames
      = trData.TracksTab, trData.DetsTab, trData.LabTab, trData.ExTab, trData.DetExTab, trData.SeqNames
    --     printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1])
    --     abort()


    return TracksTab, DetsTab, LabTab,  ExTab, DetExTab, SeqNames
  end     




  -- are we dealing with real or synthetic data
  local realData = opt.real_data == 1
  local hamidData = opt.real_data ==2


  local TracksTab, DetsTab, DetExTab, LabTab, ExTab, SeqNames  = {}, {}, {}, {}, {}, {}
  local trackModel = {}
  local VelTab = {} -- experimental

  -- learn trajectory model for synth. data
  if not realData and not hamidData then
    pm('Learning model...',2)
    local trTracksTab = 
      getTracksAndDetsTables(trainingSequences, maxTargets, maxDets, nil, false) -- (..., cropFrames, correctDets)
    trackModel = learnTrajModel(trTracksTab)
    --     for k,v in pairs(trackModel) do print(k) print(v) end
    --     abort()
    pm('   ...done',2)
  end  

  -- generate tables with sequences, detections, labels, seq-names
  if mode == 'real' then
    --     error('real?')
    local rd = opt.real_dets
    opt.real_dets = 1
    TracksTab, DetsTab, LabTab, ExTab  = 
      getTracksAndDetsTables(sequences, maxTargets, maxDets, nil, false) -- last param: correct dets
    opt.real_dets = rd
    for k,v in pairs(sequences) do table.insert(SeqNames,{v}) end
    --     printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1])
  else
    if realData then
      TracksTab, DetsTab, _, LabTab, SeqNames = 
        getRealData(nSamples, opt.mini_batch_size, trainingSequences)
    elseif hamidData then

      TracksTab, DetsTab, LabTab, ExTab, SeqNames = 
        getHamidData(nSamples,dataDir, opt)
    else
      TracksTab, DetsTab, LabTab, ExTab, SeqNames = 
        synthesizeDataY(trackModel, nSamples, opt.mini_batch_size, trainingSequences, true) -- last param = training set
    end
  end


  print(mode)

  -- add actual clutter for testing
  local dRange = getStateRange()
  if not realData and not hamidData and TESTING then
    for k,v in pairs(DetsTab) do
      for det=1,maxDets do	
        for t=1,opt.temp_win do
          if torch.rand(1):squeeze() < sopt.det_false then
            for dim=1,stateDim do
              DetsTab[k][det][t][dim] = torch.rand(1):squeeze() * dRange[3][dim] + dRange[1][dim]
            end
          end
        end
      end
    end
  end

  --   print(realData)
  --   abort()
  --   print(TracksTab)
  --   print(LabTab)
  --   print(DetsTab)
  --   abort()
  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1])
  --   abort()

  -- FLAG ALL DETECTIONS THAT REALLY EXIST
  for k,v in pairs(DetsTab) do    
    DetExTab[k] = torch.zeros(maxDets*opt.mini_batch_size, opt.temp_win):int()
    local detEx = v:narrow(3,1,1):reshape(maxDets*opt.mini_batch_size,opt.temp_win):ne(0)
    DetExTab[k][detEx] = 1
  end  
  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1])
  --   print(TracksTab)
  --   print(DetsTab)
  --  print(SeqNames)

  pm('Normalizing data...',2)
  TracksTab = normalizeData(TracksTab, DetsTab, false, opt.max_n, opt.max_m, SeqNames, singleBatch)
  DetsTab = normalizeData(DetsTab, DetsTab, false, opt.max_m, opt.max_m, SeqNames, singleBatch)  
  pm('   ...done',2)



  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1])
  --   abort()
  if opt.real_dets == 0 and opt.real_data ~= 2 and not hamidData then
    -- add noise, false and missing detections... (synthetic dets only)
    pm('Perturbing detections...',2)
    DetsTab, LabTab  = corruptDetections(TracksTab)
    pm('   ...done',2)
  end


  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  --   abort()

  -- velocities (experimental)
  if stateVel then VelTab = getVelocities(TracksTab) end

  -- transfer to GPU if needed
  for k,v in pairs(TracksTab) do TracksTab[k] = dataToGPU(TracksTab[k]) end
  for k,v in pairs(DetsTab) do DetsTab[k] = dataToGPU(DetsTab[k]) end

  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  --   abort()

  if not hamidData then  
    if not TESTING and (mode == 'train' or mode == 'validation') then
      if opt.real_dets == 0 and opt.reshuffle_dets ~= 0  then
        pm('Reshuffling... ',2)
        -- reshuffle detections and corresponding labels (synthetic dets only)
        for k,v in pairs(DetsTab) do
          local l = LabTab[k]
          v, l = reshuffleDetsAndLabels(v, l)
          LabTab[k] = l
        end
        pm('   ...done',2)
      end
    end
  end

  --   print(TracksTab, DetsTab, LabTab, ExTab, DetExTab)
  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  -- abort()





  --  print(DetsTab)
  --   printAll(TracksTab[opt.synth_training+1], DetsTab[opt.synth_training+1], LabTab[opt.synth_training+1], ExTab[opt.synth_training+1], DetExTab[opt.synth_training+1]); 
  --   abort()

  -- WARNING Adding missed detections (replace correct with clutter)



  local addMissed = true
  --   local addMissed = false
  if opt.real_dets == 1 then addMissed = false end
  if opt.real_data == 2 then addMissed = false end
  --   print(opt.real_dets)
  --   abort()
  if addMissed then

    --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
    --   abort()
    local missRate =opt.det_fail 
    local clutterRate = 0.0
    if sopt ~= nil then missRate = sopt.det_fail end -- rnnTracker (test)    
    --     print(missRate)
    --     abort()
    for k,v in pairs(DetsTab) do
      for mb = 1,opt.mini_batch_size do
        local mbStart = opt.max_n * (mb-1)+1
        local mbEnd =   opt.max_n * mb

        local mbStartD = opt.max_m * (mb-1)+1
        local mbEndD =   opt.max_m * mb

        for i=1,opt.max_n do
          local newI = mbStart+i-1
          local detRemoved = torch.ByteTensor(opt.temp_win):fill(0)
          detRemoved[1]=0
          for t=2,opt.temp_win do	
            local actualDet = LabTab[k][newI][t]
            local remPlausible = true
            if t==3 and detRemoved[2]==1 then remPlausible = false end
            if t==4 and detRemoved[3]==1 then remPlausible = false end
            if actualDet>maxDets then remPlausible = false end
            --   	  print(remPlausible, missRate, torch.rand(1):squeeze())
            if remPlausible and torch.rand(1):squeeze() < missRate then
              detRemoved[t]=1
              for dim=1,stateDim do
                -- 		print(k,mb,i,actualDet,t,dim)
                -- 		print(k,newI,t)
                DetsTab[k][actualDet+mbStartD-1][t][dim] = torch.rand(1):squeeze() * dRange[3][dim] + dRange[1][dim]
                -- 	      if torch.rand(1):squeeze() < clutterRate then DetsTab[k][actualDet][t][dim] = 0 end	      
              end
              LabTab[k][newI][t] = maxDets+1
            end
          end
        end
      end
    end

    --     abort('here')    
  end
--     printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
--     abort('here')
  -- REMOVE DETECTIONS OF INEXISTING TRACKS
  if not hamidData then
    DetsTab = removeInexistantDetections(DetsTab, ExTab, LabTab)
  end
  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  --   abort('here')




  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  --   abort()

  --- INJECT RANDOM CLUTTER
  --   if sopt == nil then
  DetsTab = injectRandomClutter(DetsTab)
  --   end
  --   TracksTab = injectRandomClutter(TracksTab, ExTab)  



  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  -- rmove spurious labels
  if not hamidData then
    for k,v in pairs(LabTab) do 
      local newLab = torch.IntTensor(1, opt.temp_win):zero()
      for mb = 1,opt.mini_batch_size do
        local mbStart = opt.max_n * (mb-1)+1
        local mbEnd =   opt.max_n * mb
        --       print(mbStart,mbEnd)      
        newLab=newLab:cat(LabTab[k][{{mbStart, mbEnd}}]:sub(1,opt.max_n), 1)
        --       print(data_batch:reshape(N,F))
        --       if mb < opt.mini_batch_size then print('---') end
      end
      newLab = newLab:sub(2,-1)
      LabTab[k] = newLab:clone()

      --     LabTab[k] = LabTab[k]:sub(1,opt.max_n) 
    end
  end
  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); 
  --   abort('here')

  -- repair broken labels (not existang = missed detection)
  if not hamidData then
    for k,v in pairs(TracksTab) do
      for i=1,maxTargets*opt.mini_batch_size do
        --       print('mb='..k..'.  sample='..i)

        local trNotExist = ExTab[k][{{i},{}}]:eq(0)
        --       print(trNotExist)
        --       print(nClasses)
        LabTab[k][i][trNotExist]=nClasses
        --       print(LabTab[k])
      end
    end
  end

  -- repair labels, flag all spurious ones as false
  for k,v in pairs(LabTab) do
    local wrongLabes = v[{{},{}}]:gt(nClasses)
    LabTab[k][wrongLabes]=nClasses
  end

  -- SMOOTH TERMINATION LABEL
  --   ExTab = smoothTerminationLabel(ExTab)


  --   if opt.real_data == 0 then printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]); end
  --   printAll(TracksTab[2], DetsTab[2], LabTab[2], ExTab[2], DetExTab[2]); 

  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]);
  --   abort('all data created')

  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]);
  --   printAll(TracksTab[2], DetsTab[2], LabTab[2], ExTab[2], DetExTab[2]);
  -- abort() 


  --   printAll(TracksTab[1], DetsTab[1], LabTab[1], ExTab[1], DetExTab[1]);
  --   abort('data')

  --   printAll(TracksTab[2], DetsTab[2], LabTab[2], ExTab[2], DetExTab[2]);




  for k,v in pairs(LabTab) do LabTab[k] = dataToGPU(LabTab[k]) end
  for k,v in pairs(VelTab) do VelTab[k] = dataToGPU(VelTab[k]) end

  -- save ?
  --   local doSave = true  
  --   if not onNetwork() then opt.doSave = true


  --   if mode=='train' then doSave = false end

  --   print(mode)
  --   print(opt.doSave)
  --   sleep(3)

  if opt.doSave and not onNetwork() then
    saveSynthTraining(opt, TracksTab, DetsTab, LabTab, ExTab, DetExTab, SeqNames, mode)
  end

  -- all done, return
  return TracksTab, DetsTab, LabTab, ExTab, DetExTab, SeqNames

end


function getHamidData(nSamples, dataDir, opt)
  --   dataDir = '/home/amilan/research/papers/2016/eccv-anton/code/RNNTracking_ECCV_2016/Track_Simulation/Data/'
  --   dataDir = './data/train/'
  dataDir = getDataDir() .. 'Synth/train/'
  local detDir = dataDir..'/det/'

  print(dataDir)
  print(detDir)
  local TracksTab, DetsTab, LabTab, ExTab, SeqNames = {},{},{},{},{}
  local nDim = 4

  local labsDefault = torch.IntTensor(maxTargets, opt.temp_win)
  for tar=1,maxTargets do labsDefault[tar] = tar end

  local exDefault = torch.IntTensor(maxTargets, opt.temp_win):fill(1)


  -- get all valid file names
  pm('Checking for file validity...')
  local minFileSize = 10
  local dataFileNames = {}
  for file in lfs.dir(dataDir) do
    local detfile = detDir..file
    local gtfile = dataDir..file
    local detfileatt = lfs.attributes(detfile)
    local gtfileatt = lfs.attributes(gtfile)
    if gtfile:find('.txt') ~= nil and lfs.attributes(detfile,'mode')~=nil  -- valid file
      and detfileatt.size>minFileSize and gtfileatt.size>minFileSize then  -- at least one line

      -- 	print(gtfile)
      local cntgt,cntdet=0,0
      local fileopen = io.open(gtfile, 'r')
      -- 	print(fileopen)

      if fileopen then for f in fileopen:lines() do cntgt=cntgt+1 break end end
      -- 	print(cntgt)
      io.close(fileopen)

      local fileopen = io.open(detfile, 'r')
      if fileopen then for f in fileopen:lines() do cntdet=cntdet+1 break end end
      io.close(fileopen)

      if cntgt>0 and cntdet>0 then	  
        table.insert(dataFileNames, file)
      else
        print('something wrong with file '..file ..'. GT: '..cntgt..'. Det: '..cntdet)
        -- 	  abort()
      end
    end	    
  end

  local nFiles = #dataFileNames
  pm('...done! '..nFiles..' valid files found.')


  local samplesRead = 0



  --   local alltracks, alldets, alllabs, allex = {},{},{},{}





  local cnt=0
  for n=1,nSamples do
    if n%math.floor(nSamples/5)==0 then 
      print((n)*(100/(nSamples))..' %...') 
    end

    local alltracks = torch.Tensor(1,opt.temp_win, stateDim)
    local alldets = torch.Tensor(1,opt.temp_win, stateDim)
    local alllabs = torch.Tensor(1,opt.temp_win):int()
    local allex = torch.Tensor(1,opt.temp_win):int()    
    local allseqnames = {}

    for m=1,opt.mini_batch_size do
      cnt=cnt+1
      if cnt>nFiles then cnt=1 end

      local file = dataFileNames[cnt]

      local detfile = detDir..file
      local gtfile = dataDir..file

      -- TRACKS
      local tracks,labs = getData(gtfile, 1)
      local dets = getData(detfile, 2)

      -- 	if tracks:nDimension()==3 and dets:nDimension()==3 and 
      -- 		tracks:size(1)>0 and tracks:size(2)>0 and
      -- 		dets:size(1)>0 and dets:size(2)>0 then


      -- 	print(file)
      if tracks:size(1)>maxTargets then tracks = tracks:narrow(1,1,maxTargets) end
      tracks = padTensor(tracks, maxTargets, 1)
      if tracks:size(2)>opt.temp_win then tracks = tracks:narrow(2,1,opt.temp_win) end
      tracks = padTensor(tracks, opt.temp_win, 2)
      tracks = tracks:narrow(3,1,stateDim)
      alltracks = alltracks:cat(tracks, 1)


      -- DETECTIONS
      --	local dets = getData(detfile, 2)
      if dets:size(1)>maxDets then dets = dets:narrow(1,1,maxDets) end
      dets = padTensor(dets, maxDets, 1)	
      if dets:size(2)>opt.temp_win then dets = dets:narrow(2,1,opt.temp_win) end
      dets = padTensor(dets, opt.temp_win, 2)	
      dets = dets:narrow(3,1,stateDim)
      alldets = alldets:cat(dets, 1)
      -- 	table.insert(alldets,dets)

      -- Labels
      if labs:size(1)>maxTargets then labs = labs:narrow(1,1,maxTargets) end
      labs = padTensor(labs, maxTargets, 1)	
      if labs:size(2)>opt.temp_win then labs = labs:narrow(2,1,opt.temp_win) end
      labs = padTensor(labs:double(), opt.temp_win, 2):int()
      labs[labs:eq(0)]=nClasses
      alllabs = alllabs:cat(labs, 1)
      -- 	table.insert(alllabs,labs)



      -- EXISTENCE
      local ex = tracks:narrow(3,1,1):reshape(maxTargets,opt.temp_win):ne(0):int()
      allex = allex:cat(ex, 1)
      -- 	table.insert(allex, ex)

      -- 	end

      --       end    
      table.insert(allseqnames, "Synth")
    end
    alltracks = alltracks:sub(2,-1)
    alldets = alldets:sub(2,-1)
    alllabs = alllabs:sub(2,-1)
    allex = allex:sub(2,-1)


    table.insert(TracksTab, alltracks)
    table.insert(DetsTab, alldets)
    table.insert(LabTab, alllabs)
    table.insert(ExTab, allex)
    table.insert(SeqNames, allseqnames)
  end

  collectgarbage()

  return TracksTab, DetsTab, LabTab, ExTab, SeqNames


end


--------------------------------------------------------------------------
--- Generate synthetic data
-- TODO: rename and docs
function synthesizeDataY(trackModels, nSynth, mbSize, trSeqTable, trainingSet)
  trainingSet = trainingSet or false
  local trTracksTab, trDetsTab, trLabTab, seqNamesTab = {}, {}, {}, {}
  local trExTab = {}

  -- try loading first
  --   print(loadSynthTraining(opt))
  --   if trainingSet and loadSynthTraining(opt) then
  --     print('Loading training data file '..opt.training_file)
  --     local trData = torch.load(opt.training_file)
  --     return trData.trTracksTab, trData.trDetsTab, trData.trLabTab, trData.trExTab, trData.seqNamesTab
  --   end

  local tt = false
  if opt.trim_tracks ~= nil and opt.trim_tracks > 0 then tt = true end
  --   print(trackModels, nSynth, mbSize)
  --   abort()
  local tL = tabLen(trackModels) 	-- how many modes?
  --   print(tL)
  --   abort()
  for n=1,nSynth do
    if n%math.floor(nSynth/5)==0 then print((n)*(100/nSynth)..' %...') end
    local alltr = torch.zeros(1, opt.temp_win, opt.state_dim)
    local alldet = torch.zeros(1, opt.temp_win, opt.state_dim)
    local alllab = torch.IntTensor(1, opt.temp_win)
    local allex = torch.IntTensor(1, opt.temp_win)
    local allseqnames = {}

    for m=1,mbSize do
      local trMode = math.random(tL)
      local seqName = trSeqTable[trMode]
      --       print(n,m,seqName)
      local trackModel = trackModels[trMode]	-- pick one model
      local imgHeight, imgWidth = imSizes[seqName]['imH'],imSizes[seqName]['imW']
      local tr = torch.zeros(1,opt.temp_win, opt.state_dim)

      local ex = torch.IntTensor(maxTargets,opt.temp_win):fill(1)
      local ntracks = torch.random(opt.max_n)      
      if opt.fixed_n ~= 0 then 
        ntracks = opt.max_n 
      end

      if ntracks<N then ex[{{ntracks+1,N},{}}]=0 end-- completely doesnt exist 
      for ss = 1,ntracks do -- at least one
        -- 	print(n,m,ss)

        local sampleTrack={}
        sampleTrack = sampleTrajectoryHamid(trackModel, opt.state_dim, imgHeight, imgWidth)  
        -- 	sampleTrack = sampleTrajectory(trackModel, opt.state_dim, imgHeight, imgWidth)

        -- 	

        local trajS = math.random(opt.temp_win/4)
        -- 	print(torch.random(opt.temp_win-3))
        local trajE = opt.temp_win - torch.random(opt.temp_win-3)+1
        -- 	print(trajE)
        -- 	sleep(.5)
        --       print(ex)
        --       print(tt)
        ----------
        -- trim tracks
        -- 	tt=false
        if tt then
          -- 	  sampleTrack[{{1},{1,trajS},{}}]=0 -- WARNING EXPERIMENT
          if trajS>1 then ex[{{ss},{1,trajS-1}}]=0 end
          if trajE-trajS>2 and trajE<opt.temp_win then
            -- 	    sampleTrack[{{1},{trajE,opt.temp_win},{}}]=0 -- WARNING EXPERIMENT
            ex[{{ss},{trajE+1,opt.temp_win}}]=0
            -- 	    local trajS1 = opt.temp_win - torch.random(opt.temp_win-3)+1
            -- 	    if trajS1>trajE and trajS1<opt.temp_win then
            -- 	      ex[{{ss},{trajS1,opt.temp_win}}]=1
            -- 	    end

          end

        end
        -- 	print(ex)
        -- 	print(trajS, trajE)
        -- 	sleep(2)
        -- 	abort()
        ----------	
        -- 	printDim(sampleTrack)

        tr = tr:cat(sampleTrack, 1)
        -- 	print()
        -- 	print(ex)


        -- 	ex[ss][sampleTrack[{{1},{},{1}}]:reshape(opt.temp_win):eq(0)] = 0


      end  
      --       print(tr)
      --       print(ex)
      --       abort()
      tr=tr:sub(2,-1) -- remove first dummy row
      tr = padTensor(tr, opt.max_n, 1)
      if tr:size(1) > opt.max_n then tr = tr:narrow(1,1,opt.max_n) end  


      --       if ntracks==2 then-- WARNING REMOVE!!!
      -- -- 	print(tr)
      -- 	local m1 = torch.mean(tr[{{1},{},{1}}])
      -- 	local m2 = torch.mean(tr[{{2},{},{1}}])
      -- -- 	print(m1,m2)
      -- 	
      -- 	if math.abs(m1-m2)<20 then
      -- 	  local flipTrack =1
      -- 	  if math.random() < 0.5 then flipTrack = 2 end  
      -- 	  tr[flipTrack] = tr[flipTrack]+100
      -- 	  
      -- 	  
      -- 	end
      --       end
      --       print(ex)
      --       abort()

      local det = tr:clone()
      --       print(det)
      --       abort()
      for ss=1,ntracks do
        for tt=1,F do
          if ex[ss][tt]==0 then 
            det[ss][tt] = 0
          end
        end
      end
      det = padTensor(det, opt.max_m, 1)
      --             print(det)
      --       abort()

      --       if opt.max_m ~= opt.max_n then error('M != N') end



      local N,F,D = getDataSize(tr)
      local lab = torch.IntTensor(N,F)
      for i=1,N do lab[i] = i end

      for i=1,N do
        local trNotExist = ex[i]:eq(0)
        lab[i][trNotExist] = nClasses

      end


      alltr = alltr:cat(tr, 1)
      alldet = alldet:cat(det, 1)
      alllab = alllab:cat(lab, 1)
      allex = allex:cat(ex, 1)
      table.insert(allseqnames, seqName)

    end
    alltr = alltr:sub(2,-1)
    alldet = alldet:sub(2,-1)
    alllab = alllab:sub(2,-1)
    allex = allex:sub(2,-1)

    table.insert(trTracksTab, alltr)
    table.insert(trDetsTab, alldet)
    table.insert(trLabTab, alllab)
    table.insert(trExTab, allex)
    table.insert(seqNamesTab, allseqnames)


  end

  --   saveSynthTraining(opt, trTracksTab, trDetsTab, trLabTab, reExTab, seqNamesTab, trainingSet)

  return trTracksTab, trDetsTab, trLabTab, trExTab, seqNamesTab
end


--------------------------------------------------------------------------
--- Generate training data by perturbing real data
function getRealData(nSynth, mbSize, trSeqTable)
  local timer=torch.Timer()
  --   local maxTargets, maxDets = 30,30 --opt.max_n, opt.max_m

  local correctDets = opt.real_dets~=0
  local sD = opt.state_dim
  stateDim, opt.state_dim = 4,4
  local cropFrames = false
--   local cropFrames = true
  local fulltrTracksTab, fulltrDetsTab, _ = 
    getTracksAndDetsTables(trSeqTable, maxTargets, maxDets, cropFrames, correctDets)


  for k,v in pairs(fulltrTracksTab) do fulltrTracksTab[k]=cleanDataTensor(fulltrTracksTab[k]) end
  for k,v in pairs(fulltrDetsTab) do fulltrDetsTab[k]=cleanDataTensor(fulltrDetsTab[k]) end
  for k,v in pairs(fulltrDetsTab) do fulltrDetsTab[k]=moveUp(fulltrDetsTab[k]) end


  local trHeights, trWidths, trueLabs, seqNamesTab = {}, {}, {}, {}

  local shiftStd = 0.0		-- standard deviation for shift (x?)
  local shiftPerc = 20 / 100		-- shift wrt. image size (uniform)
  local rotDeg = 20 / 180 * math.pi	-- rotate around image center
  local mirrorProb = .5		-- probability for mirroring in x and t
  local scrindex = {}

  local trTracksTab, trDetsTab, trOrigDetsTab, trLabTab = {}, {}, {}, {}
  for n=1,nSynth do  -- n training batches
    if n%math.floor(nSynth/5)==0 then print((n)*(100/nSynth)..' %...') end
    --     local tr = getOneBatch()

    local alltr = torch.zeros(1, opt.temp_win, stateDim)
    local alldet = torch.zeros(1, opt.temp_win, stateDim)
    local allodet = torch.zeros(1, opt.temp_win, stateDim)
    local alllab = torch.IntTensor(1, opt.temp_win):fill(opt.max_n)
    local allseqnames = {}

    for m=1,opt.mini_batch_size do
      local tL = tabLen(fulltrTracksTab)
      local randSeq = math.random(tL)	-- pick a random sequence from training set
      --       print('a',n,m)
      local randSeq, s, e = pickValidSnippet(fulltrTracksTab, fulltrDetsTab)


      local seqName = trSeqTable[randSeq]
      local imgHeight, imgWidth = imSizes[seqName]['imH'],imSizes[seqName]['imW']

      --   print(N,F,D)
      local shiftX = (torch.rand(1):squeeze()*2-1) * shiftPerc * imgWidth
      local shiftY = (torch.rand(1):squeeze()*2-1) * shiftPerc * imgHeight
      local theta = (torch.rand(1):squeeze()*2-1) * rotDeg
      local sint, cost = math.sin(theta), math.cos(theta)


      local seqTracks = fulltrTracksTab[randSeq]:narrow(2,s,opt.temp_win):clone()
      local seqDets = fulltrDetsTab[randSeq]:narrow(2,s,opt.temp_win):clone()
      --       local lab = trueLabs[randSeq]:narrow(2,s,opt.temp_win):clone()

      seqTracks = padTensor(seqTracks, opt.max_n, 1) -- this was commented out, why?
      seqDets = padTensor(seqDets, opt.max_m, 1) -- this was commented out, why?

      local N,F,D = getDataSize(seqTracks)
      local Ndet,Fdet,Ddet = getDataSize(seqDets)
      local trueTar = selectStateDim(seqTracks,1):ne(0):reshape(N,F,1):expand(N,F,D)
      local trueDet = selectStateDim(seqDets,1):ne(0):reshape(Ndet,Fdet,1):expand(Ndet,Fdet,Ddet)


      local tr = seqTracks:clone()
      local det = seqDets:clone()
      local lab = {}

      --   local tempInd = torch.linspace(s,s+opt.temp_win-1, opt.temp_win):long()  
      -- mirror in time
      local tempInd = torch.linspace(1,opt.temp_win, opt.temp_win):long()  
      local revInd = (torch.ones(opt.temp_win)*(opt.temp_win+1) - torch.linspace(1,opt.temp_win, opt.temp_win)):long()  
      if math.random() < mirrorProb then tempInd = revInd end
      local tr=tr:index(2,tempInd)
      trueTar = trueTar:index(2,tempInd)
      det=det:index(2,tempInd)
      trueDet = trueDet:index(2,tempInd)

      -- mirror in x      
      if math.random() < mirrorProb then
        -- 	print(tr)
        local N,F,D = getDataSize(tr)
        for t=1,F do for i=1,N do
          if tr[i][t][1] ~= 0 then tr[i][t][1] = imgWidth - tr[i][t][1] end
          if det[i][t][1] ~= 0 then det[i][t][1] = imgWidth - det[i][t][1] end
        end end
        -- 	print(tr)
        -- 	abort()
      end


      --     sampleTrack = sampleTrack:add(torch.randn(1):squeeze() * shiftStd)


      -- translation
      tr[{{},{},{1}}] = tr[{{},{},{1}}]:add(shiftX)
      tr[{{},{},{2}}] = tr[{{},{},{2}}]:add(shiftY)

      -- rotation (3 steps)
      -- 1. move to 0,0
      tr[{{},{},{1}}] = tr[{{},{},{1}}]:add(-imgWidth/2)
      tr[{{},{},{2}}] = tr[{{},{},{2}}]:add(-imgHeight/2)

      -- 2. rotate around image center
      for t=1,F do
        for i=1,N do
          local op = tr[i][t]:clone()      
          tr[i][t][1] = cost*op[1]-sint*op[2]
          tr[i][t][2] = sint*op[1]+cost*op[2]
        end
      end
      -- 3. move back
      tr[{{},{},{1}}] = tr[{{},{},{1}}]:add(imgWidth/2)
      tr[{{},{},{2}}] = tr[{{},{},{2}}]:add(imgHeight/2)

      tr[trueTar:eq(0)]=0
      --       tr, scrindex = reshuffleTracks(tr)


      if opt.real_dets ~= 0 then
        local N,F,D = getDataSize(seqDets)
        -- translation
        det[{{},{},{1}}] = det[{{},{},{1}}]:add(shiftX)
        det[{{},{},{2}}] = det[{{},{},{2}}]:add(shiftY)

        -- rotation (3 steps)
        -- 1. move to 0,0
        det[{{},{},{1}}] = det[{{},{},{1}}]:add(-imgWidth/2)
        det[{{},{},{2}}] = det[{{},{},{2}}]:add(-imgHeight/2)

        -- 2. rotate around image center
        for t=1,F do
          for i=1,N do
            local op = det[i][t]:clone()      
            det[i][t][1] = cost*op[1]-sint*op[2]
            det[i][t][2] = sint*op[1]+cost*op[2]
          end
        end
        -- 3. move back
        det[{{},{},{1}}] = det[{{},{},{1}}]:add(imgWidth/2)
        det[{{},{},{2}}] = det[{{},{},{2}}]:add(imgHeight/2)

        det[trueDet:eq(0)]=0

        local tt=torch.Timer()
        local Ngt,Fgt,Dgt = getDataSize(tr)
        lab = torch.IntTensor(Ngt,Fgt):fill(Ngt) -- TODO WARNING ALL FILLED WITH MAXID
        local thr=0.75
        for t=1,Fgt do
          for i=1,Ngt do
            local sDist, cTr = 1e5,1
            if sD == 4 then 
              sDist, cTr = findClosestTrackIOU(tr[i][t], det[{{},{t}}])
            else  
              sDist, cTr = findClosestTrack(tr[i][t], det[{{},{t}}])
              -- 	    print(det[{{cTr},{t}}])
              -- 	    print(torch.mean(det[{{cTr},{t},{3,4}}]))
              -- 	    abort()
              thr = torch.mean(det[{{cTr},{t},{3,4}}])
            end
            if sDist < thr then	-- associate if below threshold      
              lab[i][t] = cTr
            end	
          end      
        end

        if opt.profiler ~= 0 then  profUpdate('creating labels', tt:time().real) end 


      else -- synth dets

        local otracks = tr:clone()
        -- 	printDim(tr)
        tr, _ = reshuffleTracks(tr)
        -- 	printDim(tr)

        -- cut everything else beyond maxDets
        if tr:size(1) > opt.max_n then tr = tr:narrow(1,1,opt.max_n) end
        -- restore if too few remaining tracks
        if torch.sum(selectStateDim(tr,1):ne(0))<opt.temp_win then tr=otracks:clone() end
        if tr:size(1) > opt.max_n then tr = tr:narrow(1,1,opt.max_n) end

        det=tr:clone()
        det=padTensor(det, opt.max_m, 1)
        local N,F,D = getDataSize(tr)
        lab = torch.IntTensor(N,F)
        for i=1,N do lab[i] = i end

        -- 	printDim(det)
        -- 	abort()

        if n==1 and m==1 then 
          print('Real Data and Synthetic Detections?!?!!!')
          sleep(.5) 	  
        end
      end

      --       det=perturbDetections(det, opt.det_noise, opt.det_fail, opt.det_false) 
      allodet = allodet:cat(det, 1)


      if opt.reshuffle_dets > 0 then  det = reshuffleDets(det) end

      alltr = alltr:cat(tr, 1)
      alldet = alldet:cat(det, 1)

      --       print(lab)
      --       abort()

      alllab = alllab:cat(lab, 1)

      table.insert(allseqnames, seqName)
    end
    alltr = alltr:sub(2,-1)
    alldet = alldet:sub(2,-1)
    allodet = allodet:sub(2,-1)
    alllab = alllab:sub(2,-1)


    alltr=alltr:sub(1,-1,1,-1,1,sD)
    alldet=alldet:sub(1,-1,1,-1,1,sD)
    allodet=allodet:sub(1,-1,1,-1,1,sD)

    table.insert(trOrigDetsTab, allodet)    
    table.insert(trTracksTab, alltr)
    table.insert(trDetsTab, alldet)
    table.insert(trLabTab, alllab)
    print(trLabTab)
    table.insert(seqNamesTab, allseqnames)

  end -- for n=1,nSynth do
  if opt.profiler ~= 0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) end
  stateDim, opt.state_dim = sD,sD

  --     print(seqNamesTab)
  --     print(seqName)
  --     abort('aasd')

  return trTracksTab, trDetsTab, trOrigDetsTab, trLabTab, seqNamesTab
end  


--------------------------------------------------------------------------
--- Corrupt detection to simulate real noise and failures
--- TODO Doc
function corruptDetections(trTracksTab, valTracksTab)
  local timer=torch.Timer()
  -- WARNING DOES THIS ADD FALSE ALARMS CORRECTLY? CHECK
  --   print(trDetsTab[1][1]:t())
  local trDetsTab, trOrigDetsTab, trLabTab, trExTab = {}, {}, {}, {}
  local valDetsTab, valOrigDetsTab, valLabTab, valExTab = {}, {}, {}, {}

  for k,v in pairs(trTracksTab) do

    local alldet = torch.zeros(1, opt.temp_win, opt.state_dim)
    local allodet = torch.zeros(1, opt.temp_win, opt.state_dim)
    local allDetInd = torch.IntTensor(1, opt.temp_win)
    local allExLab = torch.IntTensor(1, opt.temp_win)

    for mb = 1,opt.mini_batch_size do      
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb

      local det = v[{{mbStart, mbEnd},{}}]:clone()
      local tr = v[{{mbStart, mbEnd},{}}]:clone()
      local faInd = {}
      --       print(v)
      --       print(mbStart, mbEnd)
      --       print(det)

      det, faInd = perturbDetections(det, opt.det_noise, opt.det_fail, opt.det_false)
      --       print(det)
      --       abort()
      allodet = allodet:cat(det:clone(), 1)

      --       table.insert(trOrigDetsTab, det)
      local N,F,D = getDataSize(tr)
      --       print(N)

      local detLab = torch.IntTensor(N,F)
      for i=1,N do 
        detLab[i]=i 
        -- 	local trNotExist = tr[{{},{},{1}}]:eq(0)
        -- 	detLab[i][trNotExist]=maxDets+1
      end

      --       if opt.max_n==opt.max_m then -- TODO
      -- 	detLab[v[{{mbStart, mbEnd},{},{1}}]:eq(0)] = opt.max_n+1 -- inexisting targets = outlier model
      --       end

      --       if opt.reshuffle_dets > 0 then 
      -- 	local scrindex={}
      -- 
      -- 	det, scrindex = reshuffleDets(det) 
      -- 	
      -- 	-- reshuffle FA like dets
      -- 	local scrInd = faInd:clone()
      -- 	local dL = detLab:clone()
      -- 	for i=1,N do for t=1,F do scrInd[{{i},{t}}] = faInd[{{scrindex[i][t]},{t}}] end end
      -- 	for i=1,N do for t=1,F do dL[{{i},{t}}] = detLab[{{scrindex[i][t]},{t}}] end end
      -- 	faInd = scrInd:clone()	
      -- 	detLab = dL:clone()
      --       end

      allDetInd = allDetInd:cat(detLab, 1)
      alldet = alldet:cat(det:clone(), 1)
      --       table.insert(trDetsTab, det)
    end

    allodet = allodet:sub(2,-1)
    alldet = alldet:sub(2,-1)
    allDetInd = allDetInd:sub(2,-1)

    --     table.insert(trOrigDetsTab, allodet)
    table.insert(trDetsTab, alldet)
    table.insert(trLabTab, allDetInd)
  end

  if opt.profiler ~= 0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) end
  --   return trDetsTab, trOrigDetsTab, valDetsTab, valOrigDetsTab, trLabTab, valLabTab
  return trDetsTab, trLabTab
end



-- TODO doc
-- randomly sample a temp. snippet to get at least 2 detections / boxes
function pickValidSnippet(fulltrTracksTab, fulltrDetsTab)
  local timer=torch.Timer()

  local tL = tabLen(fulltrTracksTab)
  assert(tL == tabLen(fulltrDetsTab), "Tables must be same length")

  -- sample according to number of dets in sequence
  local probs = torch.ones(tL) -- probabilities
  for k,v in pairs(fulltrDetsTab) do probs[k] = torch.sum(v:ne(0)) end
  probs:div(torch.sum(probs))
  --   abort()

  local minDetsInSnippet = opt.temp_win/2*opt.max_n



  local minGTInSnippet = opt.temp_win
  if opt.fixed_n ~= 0 then minGTInSnippet = opt.temp_win*opt.max_n end
  local valid = false
  local randSeq, s, e = 1, 1, opt.temp_win
  while not valid do
    randSeq = math.random(tL)	-- pick a random sequence from training set    
    randSeq = torch.multinomial(probs:float(), 1):squeeze() -- rather pick according to det density
    --     print(randSeq)
    --     sleep(.5)

    local seqTracks = fulltrTracksTab[randSeq]
    local seqDets = fulltrDetsTab[randSeq]
    local N,F,D = getDataSize(seqTracks)
    local Ndet,Fdet,Ddet = getDataSize(seqDets)
    F = math.min(F,Fdet)

    -- pick random time
    s,e = 1,opt.temp_win
    s=math.random(F-opt.temp_win)
    e=s+opt.temp_win-1  

    seqTracks = seqTracks:narrow(2,s,opt.temp_win)
    seqDets = seqDets:narrow(2,s,opt.temp_win)
    local trueTar = selectStateDim(seqTracks,1):ne(0)	-- existing detections mask
    local trueDet = selectStateDim(seqDets,1):ne(0)		-- existing target mask

    local cntTr = torch.sum(trueTar)
    local cntDet = torch.sum(trueDet) 

    local dStdOK = false        

    --     print(cntTr, cntDet)
    if cntTr>=minGTInSnippet and cntDet>=minDetsInSnippet then
      dStdOK = true
      -- check stddev
      for d=1,opt.state_dim do
        local dimDet = selectStateDim(seqTracks,d)[trueTar]
        local dStd = torch.std(dimDet)
        if dStd <= 0 then dStdOK = false; break end
        local dimDet = selectStateDim(seqDets,d)[trueDet]
        local dStd = torch.std(dimDet)
        if dStd <= 0 then dStdOK = false; break end
      end

      -- do not allow trimmed tracks --TODO: ?
      --       local trimmed = false
      --       local N,F,D = getDataSize(trueTar)
      --       -- first and last frame must exist
      --       for i=1,N do
      -- 	if trueTar[i][1] == 0 or trueTar[i][-1] == 0 then trimmed = true end
      --       end

      if dStdOK then valid = true end

    end
  end

  if opt.profiler ~= 0 then profUpdate(debug.getinfo(1,"n").name, timer:time().real) end
  return randSeq, s, e
end



--------------------------------------------------------------------------
--- Randomly remove detections to simulate detector failure.
-- @param detections	The detections tensor
-- @param failureRate	The failure rate (0,1)
-- @see perturbDetections
function removeDetections(detections, failureRate)
  failureRate = failureRate or 0.1 -- probability of detector failure [0,1]

  local ldet = detections:clone()
  local odet = ldet:clone()

  local N,F,D = getDataSize(detections)

  local mask = torch.rand(N,F,1)
  mask = torch.le(mask,failureRate) -- binary mask to remove
  mask = mask:reshape(N,F,1):expand(N,F,D) -- expand to all dims

  ldet:maskedFill(mask,opt.dummy_det_val)

  -- preserve first frame! --WARNING
  for i=1,N do ldet[i][1] = odet[i][1] end

  return ldet

end

--------------------------------------------------------------------------
--- Make artificial detections more realistic. Add noise, remove dets, add false alarms
-- @param detections	The detections tensor
-- @param noiseVar		The noise variance multiplier
-- @param failureRate	The failure rate (0,1)
-- @param falseRate	The false alarm rate (empty spots in tensor are filled with this probability)
function perturbDetections(detections, noiseVar, failureRate, falseRate)

  noiseVar = noiseVar or 0.01
  failureRate = failureRate or 0.2
  falseRate = falseRate or 0.1

  local N,F,D = getDataSize(detections)

  local zeroMask = torch.eq(detections, 0)	-- find inexistent detections
  --   print(N,F,D)
  --   print(detections)
  --   print(noiseVar)
  detections = detections + torch.randn(N,F,D) * noiseVar
  detections[zeroMask] = 0
  --   detections = removeDetections(detections, failureRate)
  detections = padTensor(detections, opt.max_m, 1)
  local N,F,D = getDataSize(detections)

  -- det range

  local dRange = torch.zeros(3,D)
  for d=1,D do
    local detDimSlice = selectStateDim(detections, d)
    if torch.sum(detDimSlice[detDimSlice:ne(0)]) == 0 then
      dRange[1][d]=-.5
      dRange[2][d]=.5
      dRange[3][d]=1      
    else
      dRange[1][d] = torch.min(detDimSlice[detDimSlice:ne(0)])
      dRange[2][d] = torch.max(detDimSlice[detDimSlice:ne(0)])
      dRange[3][d] = dRange[2][d]-dRange[1][d]
    end
  end

  -- override
  -- fix range to uniformly fill image space
  for d=1,D do
    dRange[1][d]=-.5
    dRange[2][d]=.5
    dRange[3][d]=1  
  end
  --   print(dRange)
  --   abort()

  -- this is most likely inefficient
  local faInd = torch.ByteTensor(N,F):fill(0) -- false alarm index
  --   print(dRange)
  --   printDim(detections)
  --   printDim(detections,2)
  for t=1,F do
    for i=opt.max_n+1,N do
      if detections[i][t][1] == 0 then
        if math.random() < falseRate then
          for d=1,D do 
            detections[i][t][d] = torch.rand(1):squeeze() * dRange[3][d] + dRange[1][d]
          end
          faInd[i][t] = 1
        end
      end
    end
  end
  --   printDim(detections)
  --   printDim(detections,2)  
  --   abort()

  return detections, faInd
end



--------------------------------------------------------------------------
--- Returns ground truth and detections for a set of sequences
-- @param seqTable	The table of sequences to be read
-- @param maxTargets	Max number of targets (per frame) to pad tensors
-- @param maxDets	Max number of detections (per frame) to pad tensors
-- @param cropFrames	Whether to trim the sequence to a time window
function getTracksAndDetsTables(seqTable, maxTargets, maxDets, cropFrames, correctDets,detfile)

  if cropFrames == nil then cropFrames = true end

  --   local correctDets = false
  if correctDets == nil then correctDets = false end  
  --   print(cropFrames)
  --   print(correctDets)

  --   error('implement ExTab')

  local allTracksTab = {}
  local allDetsTab = {}  
  local allLabTab = {}
  local allExTab = {}
  local allDetExTab = {}
  --   if opt.real_dets ~= 0 then error('REAL LABELS!!1!') end
  for _, seqName in pairs(seqTable) do
    local tracks = getGTTracks(seqName)
    local detections = getDetTensor(seqName,detfile)
    --     print(torch.sum(detections:ne(0)))
    --     abort('tad')

    local labels = {}
    if opt.real_dets == 0 then detections = tracks:clone() end


    local tracksOrig = tracks:clone()
    local detsOrig = detections:clone()
    local meanWidth = getMeanDetWidth(detections)
    --     print(meanWidth)

    if stateDim ~= nil then 
      tracks=tracks:sub(1,-1,1,-1,1,stateDim)
      detections=detections:sub(1,-1,1,-1,1,stateDim)
    end

    --     printDim(detections) 
    local ts = 1 -- first frame
    if sopt~=nil and sopt.fframe ~= nil then ts = sopt.fframe end
    if opt ~= nil and cropFrames then
      local F = opt.temp_win -- subseq length
      tracks = selectFrames(tracks,ts,ts+F-1)
      detections = selectFrames(detections,ts,ts+F-1)
    end
    -- printDim(detections) abort('det')
    -- pad tensors with zeros if necessary
    tracks = padTensor(tracks, maxTargets, 1)
    detections = padTensor(detections, maxDets, 1)
    --     print(maxTargets)
    --     abort()

    -- if synthetic detections, try random tracks
    if opt.real_dets == 0 then
    --       tracks = reshuffleTracks(tracks)
    end


    -- trim tracks to maxTargets if necessary
    if tracks:size(1) > maxTargets then tracks = tracks:narrow(1,1,maxTargets) end


    local Ngt,Fgt,Dgt = getDataSize(tracks)
    local Ndet,Fdet,Ddet = getDataSize(detections)

    local thr = meanWidth/2
    --     print(correctDets)
    if correctDets then

      local corDets= detections:clone()
      for t=1,Fgt do
        for i=1,Ngt do
          local sDist, cTr = findClosestTrack(tracks[i][t], detections[{{},{t}}])
          -- 	  print(i,t,sDist, thr)
          -- 	  if sDist > thr then	-- associate if below threshold      
          -- 	    detections[{{cTr},{t},{}}] = 0
          -- 	  end	
          if sDist < thr then 
            local a,b = detections[i][t]:clone(), detections[cTr][t]:clone()
            detections[i][t]=b
            detections[cTr][t]=a
          end
        end
      end

      for t=1,Fdet do
        for i=1,Ndet  do
          local sDist, cTr = findClosestTrack(detections[i][t], tracks[{{},{t}}])
          if sDist > thr then	-- associate if below threshold      
            detections[i][t] = 0
          end	
        end
      end

      -- cut everything else beyond maxDets
      if detections:size(1) > maxDets then detections = detections:narrow(1,1,maxDets) end

      -- keep first frame
      local detsO = detections:clone()
      detections=reshuffleDets(detections)
      detections[{{},{1},{}}]=detsO[{{},{1},{}}]:clone()
    end

    -- now trim dets

    if detections:size(1) > maxDets then detections = detections:narrow(1,1,maxDets) end
    if opt.reshuffle_dets~=0 then detections = reshuffleDets(detections) end

    local Ngt,Fgt,Dgt = getDataSize(tracks)
    local Ndet,Fdet,Ddet = getDataSize(detections)

    table.insert(allTracksTab, tracks)
    table.insert(allDetsTab, detections)

    -- TODO do all this properly!!!
    if opt.real_dets ~= 0 then
      labels = torch.IntTensor(Ngt,Fgt):fill(Ngt) -- TODO WARNING ALL FILLED WITH MAXID
      --       thr=0.5
      for t=1,Fgt do
        for i=1,Ngt do
          local sDist, cTr = findClosestTrack(tracks[i][t], detections[{{},{t}}])
          -- 	  print(i,t)
          -- 	  print(sDist, cTr, thr)
          -- 	  sleep(2)
          if sDist < thr then	-- associate if below threshold      
            labels[i][t] = cTr
          end	
        end      
      end

    else
      local N,F,D = getDataSize(detections)
      labels = torch.IntTensor(N,F)
      for i=1,N do labels[i] = i end
    end

    local ex = torch.IntTensor(maxTargets, opt.temp_win):fill(1)
    --     print(tracks)
    for i=1,Ngt do
--             print(tracks)
--             print(opt.temp_win)
  print(opt.temp_win)
            print(tracks[{{i},{},{1}}]:size())
      local trNotExist = tracks[{{i},{},{1}}]:reshape(opt.temp_win):eq(0)
      ex[i][trNotExist] = 0
      labels[i][trNotExist] = maxDets+1
    end
    --     print(tracks)   
    --     print(labels)
    --     abort()

    table.insert(allLabTab, labels)
    table.insert(allExTab, ex)
  end

  --   print(allTracksTab)
  --   print(allLabTab)
  --   abort()

  return allTracksTab, allDetsTab, allLabTab, allExTab
end




--------------------------------------------------------------------------
--- Get one single sample for one trajectory from model m
function sampleTrajectory(m, stateDim, imH, imW)

  local validTraj = false

  while not validTraj do
    validTraj = true
    -- create zero track of length and dim
    local trZ = torch.zeros(1, opt.temp_win, stateDim)


    -- sample start / end frame
    local featInd = 1
    --   print()
    local s = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    s=1
    --     if s<1 then return trZ end

    featInd = 2;
    local e = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    e=opt.temp_win
    --     if e>opt.temp_win then return trZ end  
    if e>opt.temp_win then validTraj = false end
    local F = e-s+1
    --     if F<=0 then return trZ end
    if F<=0 then validTraj = false end


    local tr = torch.zeros(opt.temp_win, stateDim)

    for d = 1,stateDim do
      -- get linear sample
      featInd = 3+2*(d-1)
      --     print(d, m[1][featInd])
      local sx = sampleNormal(m[1][featInd], m[2][featInd])

      featInd = 3 + 2*stateDim + (d-1)
      --     print(featInd)
      --     print(m)
      local mv = sampleNormal(m[1][featInd], m[2][featInd])
      --       print(mv)
      --       mv = mv + torch.randn(1):squeeze()*1
      --       print(mv)
      local ex = sx + mv * F
      --       print(mv)
      --       print(sx)
      --       abort()
      --     if sx>ex then sx,ex = ex,sx end -- this might be a bug! WARNING
      local velNoise = .01
      for t=s,e do tr[t][d] = sx + (mv + torch.randn(1):squeeze()*velNoise) * (t-s) + torch.randn(1):squeeze()*2  end

      --       print(tr)
      --       abort()
      -- MAKE TURN HACK
      if torch.rand(1):squeeze()<0.5 then
        local mv2 = sampleNormal(m[1][featInd], m[2][featInd])
        -- 	print(mv,mv2)
        s = torch.random(opt.temp_win - 3)
        -- 	s = opt.temp_win - math.random(opt.temp_win/2)
        for d=1,stateDim do
          sx = tr[s][d]
          for t=s,e do tr[t][d] = sx + (mv2 + torch.randn(1):squeeze()*velNoise) * (t-s) + torch.randn(1):squeeze()*2  end
        end
      end
      --       print(tr)
      --       sleep(2)
    end
    --   end

    --   print(tr)


    -- remove invalid parts

    local F,D = getDataSize(tr)
    local valids = torch.ByteTensor(F,D):fill(1)

    -- these are absolute
    local minX,maxX = 1,imW
    local minY,maxY = 1,imH
    local minW,maxW = 1,imW -- TODO
    local minH,maxH = 1,imH -- TODO
    local minD=torch.Tensor({minX, minY, minW, minH})
    local maxD=torch.Tensor({maxX, maxY, maxW, maxH})
    for t=1,F do
      for d=1,D do
        local x = tr[t][d]
        if x < minD[d]  or x > maxD[d] then valids[t] = 0 end
      end
    end
    --     print(valids)
    tr[valids:eq(0)] = 0

    --   print(tr)
    --   abort()  
    if torch.sum(valids)<opt.temp_win/2 then validTraj = false end
    if torch.sum(valids)<opt.state_dim*opt.temp_win then validTraj = false end

    -- force to start and end inside image
    --     if tr[1][1] == 0 or tr[opt.temp_win][1] == 0 then validTraj = false end

  end
  --   print(tr)
  --   sleep(1)
  return tr:reshape(1,opt.temp_win, stateDim)


end

function getTransMat(T)
  local mat = torch.eye(stateDim*2)
  for d=1,stateDim*2,2 do
    mat[d][d+1] = T
  end
  return mat
end


function getNoiseCov(T)
  local T3=T*T*T/3
  local T2=T*T/2

  local mat = torch.Tensor({{T3,T2},{T2,T}})
  if stateDim==2 then
    mat = torch.Tensor({
      {T3, T2,  0,  0},
      {T2,  T,  0,  0},
      {0,   0, T3, T2},
      {0,   0, T2,  T}})
  elseif stateDim==3 then
    mat = torch.Tensor({
      {T3, T2,  0,  0,  0,  0},
      {T2,  T,  0,  0,  0,  0},
      {0,   0, T3, T2,  0,  0},
      {0,   0, T2,  T,  0,  0},
      {0,   0,  0,  0, T3, T2},
      {0,   0,  0,  0, T2,  T}})
  elseif stateDim==4 then
    mat = torch.Tensor({
      {T3, T2,  0,  0,  0,  0,  0,  0},
      {T2,  T,  0,  0,  0,  0,  0,  0},
      {0,   0, T3, T2,  0,  0,  0,  0},
      {0,   0, T2,  T,  0,  0,  0,  0},
      {0,   0,  0,  0, T3, T2,  0,  0},
      {0,   0,  0,  0, T2,  T,  0,  0},
      {0,   0,  0,  0,  0,  0, T3, T2},
      {0,   0,  0,  0,  0,  0, T2,  T}})
  end
  return mat  
end


--------------------------------------------------------------------------
--- Get one single sample for one trajectory from model m
function sampleTrajectoryHamid(m, stateDim, imH, imW)

  local validTraj = false

  while not validTraj do
    validTraj = true
    -- create zero track of length and dim
    local trZ = torch.zeros(1, opt.temp_win, stateDim)


    -- sample start / end frame
    local featInd = 1
    --   print()
    local s = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    s=1
    --     if s<1 then return trZ end

    featInd = 2;
    local e = torch.round(sampleNormal(m[1][featInd], m[2][featInd]))
    e=opt.temp_win
    --     if e>opt.temp_win then return trZ end  
    if e>opt.temp_win then 
      validTraj = false 
      --       print('e>opt.temp_win'..e) 
    end
    local F = e-s+1
    --     if F<=0 then return trZ end
    if F<=0 then 
      validTraj = false 
      --       print('F<=0'..F) 
    end


    tr = torch.zeros(opt.temp_win, stateDim)

    local velNoise = 2
    local T =1
    local noiseMean = torch.zeros(stateDim*2)
    local Q = getNoiseCov(T)    
    local eta = distributions.mvn.rnd(noiseMean, Q) * velNoise
    local transMat = getTransMat(T)


    local fullVec = torch.zeros(stateDim*2,1):float()

    for d = 1,stateDim do
      -- get linear sample
      featInd = 3+2*(d-1)
      local sx = sampleNormal(m[1][featInd], m[2][featInd])

      featInd = 3 + 2*stateDim + (d-1)
      local mv = sampleNormal(m[1][featInd], m[2][featInd])
      local ex = sx + mv * F
      fullVec[2*d-1][1] = sx
      fullVec[2*d][1] = mv      
    end

    --     print(fullVec)
    --     abort()
    for d=1,stateDim do tr[1][d] = fullVec[2*d-1] end

    --     print(fullVec)
    for t=s+1,e do
      eta = distributions.mvn.rnd(noiseMean, Q) * velNoise
      fullVec = torch.mm(transMat:double(), fullVec:double()):float() + eta:float()
      --       print(fullVec)
      for d=1,stateDim do tr[t][d] = fullVec[2*d-1] end
    end
    --     abort()


    --   print(tr)


    -- remove invalid parts

    local F,D = getDataSize(tr)
    local valids = torch.ByteTensor(F,D):fill(1)

    -- these are absolute
    --     print(imW,imH)
    local minX,maxX = 1,imW
    local minY,maxY = 1,imH
    local minW,maxW = 1,imW -- TODO
    local minH,maxH = 1,imH -- TODO
    local minD=torch.Tensor({minX, minY, minW, minH})
    local maxD=torch.Tensor({maxX, maxY, maxW, maxH})
    --     print(valids)
    for t=1,F do
      for d=1,D do
        local x = tr[t][d]
        if x < minD[d]  or x > maxD[d] then valids[t] = 0 end
      end
    end
    --     print(valids)
    --     print(tr)
    --     tr[valids:eq(0)] = 0

    --   print(tr)
    -- --   abort()  
    --     print(torch.sum(valids))
    --     print(tr)
    --     print(s,e)
    --     sleep(1)
    if torch.sum(valids)<opt.temp_win/2 then 
      validTraj = false 
      --       print('sum valids'..torch.sum(valids)) 
    end
    --     if torch.sum(valids)<opt.state_dim*opt.temp_win then validTraj = false end

    -- force to start and end inside image
    --     if tr[1][1] == 0 or tr[opt.temp_win][1] == 0 then validTraj = false end

  end
  --   print(tr)
  --   sleep(1)
  return tr:reshape(1,opt.temp_win, stateDim)


end

--------------------------------------------------------------------------
--- Normalize data to 0-mean 
-- @param 	n is the number of targes/measurements per mini batch
function normalizeData(dataTab, detTab, backwards, n, m, seqNamesTab, singleBatch)
  backwards = backwards or false
  singleBatch = singleBatch or false

  assert(#dataTab == #detTab, 'tables are different lengths')

  local shiftTab, divTab = {},{}
  local normImSize = false
  if opt.norm_std ~= nil and opt.norm_std < 0 then normImSize = true end

  if normImSize then assert(#dataTab == #seqNamesTab, 'tables are different lengths') end

  --   local opt.mini_batch_size = dataTab[1]:size(1)/n
--   local opt.mini_batch_size=opt.mini_batch_size
  if singleBatch then opt.mini_batch_size = 1 end
  --   if perBatch then opt.mini_batch_size = opt.mini_batch_size end
  for k,v in pairs(dataTab) do
    shiftTab[k] = {}
    divTab[k] = {}
    for mb = 1,opt.mini_batch_size do
      shiftTab[k][mb] = {}
      divTab[k][mb] = {}

      -- batch pointers for tracks
      local mbStart = n * (mb-1)+1
      local mbEnd =   n * mb

      -- batch pointers for detections
      local mbDetStart = m * (mb-1)+1
      local mbDetEnd =   m * mb

      local detsInBatch = detTab[k]:clone()			-- N*mbxFxD

      local N,F,D = getDataSize(v)
      local Ndet,Fdet,Ddet = getDataSize(detsInBatch)
      if singleBatch then 
        mbStart, mbEnd = 1,N
        mbDetStart, mbDetEnd = 1,Ndet
      end


      detsInBatch = detsInBatch[{{mbDetStart, mbDetEnd},{}}]
      --     local vloc = v:clone()
      local vloc = v[{{mbStart, mbEnd},{}}]

      local trueDet = selectStateDim(detsInBatch,1):ne(0)	-- existing detections mask
      local trueTar = selectStateDim(vloc,1):ne(0)		-- existing target mask
      if torch.sum(trueDet) > 1 then 			-- handle special case <= 1 det
        local dMean = torch.zeros(stateDim)		-- mean vector per dimension
        local dStd = torch.ones(stateDim)			-- std vector

        for d=1,stateDim do
          local dimDet = selectStateDim(detsInBatch,d)[trueDet]    


          local divFactor = torch.std(dimDet)
          local shiftFactor = torch.mean(dimDet)
          if opt.norm_std ~= nil and opt.norm_std > 0 then divFactor = opt.norm_std 
          elseif normImSize then
            -- 	  print(k,mb)
            local seqName = seqNamesTab[k][mb]
            -- 	  print(seqName)
            if d%2 == 0 then divFactor = imSizes[seqName]['imH']
            else divFactor = imSizes[seqName]['imW']
            end
          end


          if opt.norm_mean ~= nil and opt.norm_mean == 0 then shiftFactor = 0 end

          dStd[d] = divFactor
          if normImSize then shiftFactor = divFactor/2 end
          dMean[d] = shiftFactor


          shiftTab[k][mb][d] = shiftFactor
          divTab[k][mb][d] = divFactor


          -- 	assert(dStd[d]>0, string.format('std. dev. not positive: %f', dStd[d]))
          if dStd[d]<=0 then pm(string.format('std. dev. not positive: %f', dStd[d]),3) end
        end


        for d=1,stateDim do
          -- 	print(vloc[{{},{},{d}}]:reshape(1,50))
          if dStd[d]>0 then
            -- 	  print(d,dStd[d], dMean[d], vloc[{{},{},{d}}])
            if backwards then
              vloc[{{},{},{d}}] = vloc[{{},{},{d}}] * dStd[d] + dMean[d]
            else
              vloc[{{},{},{d}}] = (vloc[{{},{},{d}}] - dMean[d]) / dStd[d]            
            end
          end
          -- 	print(vloc[{{},{},{d}}]:reshape(1,50))
        end

        -- reset inexisting ones to 0
        --       local N,F,D = getDataSize(detsInBatch)
        --       for i=1,N do
        -- 	for t=1,F do
        -- 	  if trueTar[i][t]==0 then v[i][t] = 0 end
        -- 	end
        --       end
        -- vectorized implementation
        -- newTT = trueTar:repeatTensor(1,1,2):squeeze():t():reshape(2,3,2)
        local N,F,D = getDataSize(vloc)
        local newTT = trueTar:reshape(N,F,1):expand(N,F,D)
        vloc[newTT:eq(0)]=0      
      else
      --       print(detsInBatch)
      --       pm('WARNING! No detections in batch!!!',1)
      --       abort('aa')
      end

      --     print(k)

      --     abort('ndata')
    end
  end
  return dataTab, shiftTab, divTab
end


--------------------------------------------------------------------------
--- show samples of tracks 
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
    
    local plotTab = getTrackPlotTab(tracks:sub(1,maxTargets):float(), {}, 1)
    plotTab = getDetectionsPlotTab(detections:sub(1,maxDets):float(), plotTab, nil, da)

    print(seqName[1], seq)
    plot(plotTab, winID, string.format('%s-%s-%06d','Traning-Data',seqName[1], seq), nil, 1) -- do not save first
    sleep(1)
  end

end