--------------------------------------------------------------------------
--- Prints column names for process table
function printTrainingHeadline()
  local headline = 
    string.format("%14s%10s%8s%9s%8s%6s%6s","Iter.","Tr. loss","G-norm","tm/btch","l-rate","ETL","TELP")
  print(headline)
end

--------------------------------------------------------------------------
--- Prints numbers for current iteration
-- @param sec 	Seconds passed since start
function printTrainingStats(i, me, tl, gn, t, lr, sec)
  local secLeft = (sec / (i/opt.max_epochs) - sec)
  local hrsLeft = math.floor(secLeft / 3600)
  local minLeft = torch.round((secLeft % 3600)/60)
  local hrsElapsed = math.floor(sec / 3600)
  local minElapsed = torch.round((sec % 3600)/60)
  
  print(string.format("%6d/%7d%10.5f %.1e%8.2fs %.1e%3d:%02d%3d:%02d", i, me, tl, gn, t,lr, hrsLeft,minLeft,hrsElapsed,minElapsed))
end


--------------------------------------------------------------------------
--- Model-specific options in a line
function printModelOptions(opt)
  local modelParams = opt.modelParams
  local header = ''
  local params = ''
  for k,v in pairs(modelParams) do 
    if string.len(v) > 5 then v = string.sub(v,1,5) end
    header = header..string.format('%6s',v) 
  end
  for k,v in pairs(modelParams) do params = params..string.format('%6d',opt[v]) end
  print(header)
  print(params)
end

function printDim(data, dim)
  dim=dim or 1
  local N,F,D = getDataSize(data)
  if opt.mini_batch_size == 1 then
    print(data:narrow(3,dim,1):reshape(N,F))
  else
    N = N / opt.mini_batch_size
    for mb = 1,opt.mini_batch_size do
      local mbStart = N * (mb-1)+1
      local mbEnd =   N * mb
      local data_batch = data[{{mbStart, mbEnd}}]:clone()
      print(data_batch:narrow(3,dim,1):reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end
end

function printAll(tr, det, lab, ex, detex)
  local N,F,D = getDataSize(tr)
  
  if lab=={} then lab=nil end
  if ex=={} then lab=nil end
  if detex=={} then lab=nil end
  
  local dim = 1
  print('--------   Tracks   -----------')
  printDim(tr, dim)
  
  print('-------- Detections -----------')
  printDim(det, dim)
  
  
  local N = lab:size(1)/opt.mini_batch_size
  if lab~= nil then 
  print('--------   Labels   -----------')
--   print(lab)  
  for mb = 1,opt.mini_batch_size do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb
    local data_batch = lab[{{mbStart, mbEnd}}]:clone()
--     print(data_batch)
--     print(N,F)
    print(data_batch:reshape(N,F))
    if mb < opt.mini_batch_size then print('---') end
  end     
  end
    
  
  if ex~= nil then 
    local N = ex:size(1)/opt.mini_batch_size
    print('--------  Existance -----------')    
    for mb = 1,opt.mini_batch_size do
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb
      local data_batch = ex[{{mbStart, mbEnd}}]:clone()
--     print(data_batch)
--     print(N,F)      
      print(data_batch:reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end
    
  if detex~= nil then
    local N = detex:size(1)/opt.mini_batch_size
    print('------- Det. Existance --------')
    for mb = 1,opt.mini_batch_size do
      local mbStart = N * (mb-1)+1
      local mbEnd =   N * mb
      local data_batch = detex[{{mbStart, mbEnd}}]:clone()
--       print(data_batch)
--       print(N,F)
      print(data_batch:reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end

  
  print('-------------------------------')
end
