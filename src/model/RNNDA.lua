--[[ 
  An RNN for predicting data association
]]--

require 'model.ticks'

local RNN = {}

function RNN.rnn(opt)
  
  local RNNMODE, LSTMMODE, GRUMODE = getRNNMode(opt)
  
--   assert(false)
--   print('aa')
  -- set default parameter values
  local stateDim = opt.state_dim
  local maxTargets = opt.max_n
  if opt.max_nf ~= nil then maxTargets = opt.max_n+opt.max_nf end
  local maxDets = opt.max_m
  local b = opt.batch_size
  local rnnSize = opt.rnn_size
  
  local dropout = opt.dropout or 0 
  local n = opt.num_layers or 1 	-- number of layers  
  local linp = opt.linp~= 0	  	-- DA as input
  local einp = opt.einp~= 0		-- Ex as input
  local vel = opt.vel~= 0		-- velocity in state
  local updLoss = opt.kappa ~= 0
  local predLoss = opt.lambda ~= 0
  local daLoss = opt.mu ~= 0

  local exvar = opt.nu ~= 0
  local onehot = opt.one_hot ~= 0
-- if exVar then error('n  
  -- inputs  
  local inputs = {}
  local xSize = stateDim*maxTargets
  if vel then xSize = stateDim*maxTargets*2 end
  local dSize = stateDim*maxDets
  
  -- number of hidden inputs (1 for RNN, 2 for LSTM)
  local nHiddenInputs = 1
  if LSTMMODE then nHiddenInputs = 2 end
  
  local detInpInd = n*nHiddenInputs+1
  local labInpInd = n*nHiddenInputs+2
  local exInpInd = n*nHiddenInputs+2
  if linp then detInpInd = detInpInd+1 end
  if einp then detInpInd = detInpInd+1 end
--   local labInpInd
--   print(xSize)
--   print(labInpInd)
  
  local nMissed =0 -- 0 or 1
  local nClasses = maxDets+1

  local batchMode = opt.mini_batch_size>1
  

  
  
  -- input for pairwise distances
    table.insert(inputs, nn.Identity()():annotate{
    name='PW', 
    description='PW Dist',
    graphAttributes = {color='red', style='filled'}
    })
  
  
  for L = 1,n do
    -- one input for previous location
    if LSTMMODE then table.insert(inputs, nn.Identity()():annotate{name='c^'..(L)..'_t'}) end
    table.insert(inputs, nn.Identity()():annotate{name='h^'..(L)..'_t'})    
  end

  
  
  
  local inSize = maxTargets*maxDets
  if opt.pwd_mode ==1 then inSize = maxTargets*maxDets*math.min(2,stateDim) end
  
  local x, prev_d, stateDimL, xl, exi, inputSizeL
  local outputs = {}
  local pred_state = {} 
    
  local labToH = {}
--   
  -----------------------
  -- DATA  ASSOCIATION --
  -----------------------
  local DA_state = {}
  -- connect top hidden output to new hidden input
  
  for L=1, n do
    -- hidden input from previous RNN
    local prev_h = inputs[nHiddenInputs*L + 1]
    local prev_c = {}
    
    if LSTMMODE then prev_c = inputs[2*L] end
    
    -- real input
    if L==1 then
      x = inputs[1]
      inputSizeL = inSize
    else      
      x = DA_state[(L-1)*nHiddenInputs]
      inputSizeL = rnnSize
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end
        
    -- Do one time tick
    local next_c, next_h = {}, {}
    if RNNMODE then DA_state = RNNTick(DA_state, opt, inputSizeL, x, prev_h, L) end
    if LSTMMODE then DA_state = LSTMTick(DA_state, opt, inputSizeL, x, prev_h, prev_c, L) end

  end -- layers
  
  
  
  -- set up the decoder for data association
  local top_DA_state = DA_state[#DA_state]
  

  
  if dropout > 0 then top_DA_state = nn.Dropout(dropout)(top_DA_state) end
  local da = nn.Linear(rnnSize, (maxTargets)*(nClasses))(top_DA_state):annotate{
      name='DA_t',
      description='data assoc.',
      graphAttributes = {color='green', style='filled'}
    }

    
--   localDaRes = nn.Reshape(opt.max_n, nClasses, batchMode)(da):annotate{name='Rshp DA'}
  local localDaRes = nn.Reshape(opt.max_n*nClasses, batchMode)(da):annotate{name='Rshp DA'}
  
  local daFinal = localDaRes
  if opt.bce == 0 then
--     daFinal = nn.LogSoftMax()(localDaRes):annotate{
--       name='DA_t',
--       description='data assoc. LogSoftMax',
--       graphAttributes = {color='green'}
--     }

    daFinal = nn.Sigmoid()(localDaRes):annotate{
      name='DA_t',
      description='data assoc. Sigmoid',
      graphAttributes = {color='green'}            
    }    
  else    
    daFinal = nn.Sigmoid()(localDaRes):annotate{
      name='DA_t',
      description='data assoc. Sigmoid',
      graphAttributes = {color='green'}            
    }
--     daFinal = nn.LogSoftMax()(localDaRes):annotate{
--       name='DA_t',
--       description='data assoc. LogSoftMax',
--       graphAttributes = {color='green'}
--     }
    
  end
    

  -- insert hidden states to output
  for _,v in pairs(DA_state) do table.insert(outputs, v) end

  
  if daLoss then table.insert(outputs,daFinal) end
--   if daLoss then table.insert(outputs,nn.Identity()(daFinal)) end
  
  -- one-to-one loss
  if opt.bce~=0 then 
    local vecReshape = nn.Reshape(opt.max_n, nClasses, batchMode)(nn.Exp()(daFinal))    
    if batchMode then
      table.insert(outputs, nn.Sum(3, 2)(vecReshape)) 
      table.insert(outputs, nn.Sum(2, 2)(nn.Narrow(3,1,opt.max_n)(vecReshape))) -- <= each detection should not be assigned more than once
    else
      table.insert(outputs, nn.Sum(2, 2)(vecReshape)) -- == each target should be assigned once
      table.insert(outputs, nn.Sum(1, 2)(nn.Narrow(2,1,opt.max_n)(vecReshape))) -- <= each detection should not be assigned more than once
    end
  end


  return nn.gModule(inputs, outputs)  
  
end
return RNN