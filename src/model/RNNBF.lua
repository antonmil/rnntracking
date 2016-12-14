--[[
  An RNN that emulates a Bayesian Filter

]]--

require 'model.ticks';

local RNN = {}

function RNN.rnn(opt)
  local RNNMODE, LSTMMODE, GRUMODE = getRNNMode(opt)

  -- set default parameter values
  local stateDim = opt.state_dim
--  local opt.max_n = opt.max_n
  if opt.max_nf ~= nil then opt.max_n = opt.max_n+opt.max_nf end
  local maxDets = opt.max_m
  local b = 1 --opt.batch_size
  local rnnSize = opt.rnn_size

  local dropout = opt.dropout or 0
  local n = opt.num_layers or 1 	-- number of layers
  local linp = opt.linp~= 0	  	-- DA as input
  local einp = opt.einp~= 0		-- Ex as input
  local vel = false		-- velocity in state
  local predLoss = opt.lambda ~= 0
  local daLoss = opt.mu ~= 0
  local exLoss = opt.nu ~= 0
  local exvar = opt.nu ~= 0
  local exSmoothLoss = opt.xi ~= 0
  local dynSmoothLoss = false
  local onehot = opt.one_hot ~= 0
  -- if exVar then error('n
  -- inputs
  local inputs = {}
--  local opt.xSize = stateDim*maxTargets
--  if vel then opt.xSize = stateDim*maxTargets*2 end
--  local opt.dSize = stateDim*maxDets

  -- number of hidden inputs (1 for RNN, 2 for LSTM)
--  local opt.nHiddenInputs = 1
  if LSTMMODE then opt.nHiddenInputs = 2 end

  local stateInpInd = 1
  local detInpInd = n*opt.nHiddenInputs+2
  local labInpInd = detInpInd+1
  local exInpInd = labInpInd+1
  --   print(exInpInd)
  --   if linp then detInpInd = detInpInd+1 end
  --   if einp then detInpInd = detInpInd+1 end

  --   print(xSize)

  local miniBatchSize = opt.mini_batch_size
  local batchMode = opt.mini_batch_size>1
  batchMode = true
  local nClasses = maxDets+1

  print('--- INPUT  INDICES ---')
  print(string.format('%8s%8s%8s%8s','X','det','lab','ex'))
  print(string.format('%8d%8d%8d%8d',stateInpInd,detInpInd,labInpInd,exInpInd))



  local h1Size, h2Size = rnnSize, rnnSize
  local splitModel = false  --opt.model_split ~= 0
  if splitModel then
    error('split hidden state not quite done')
    h1Size = torch.round(rnnSize/opt.model_split)
    h2Size = rnnSize - h1Size
  end

  -- auxiliary index
  --   local ind = torch.LongTensor(maxTargets*stateDim)
  --   local cnt = 0
  --   for tar=1,maxTargets do
  --     for d=1,stateDim do
  --       local offset = (tar-1) * maxTargets*stateDim + d
  --       cnt=cnt+1
  --       ind[cnt] = offset
  --     end
  --   end
  --   print(ind)
  --   abort()
  --

  --
  function doDP(opt, lab, det, inp)
    --     function doDPDebug(opt, lab, det, inp)
    local maxTargets, maxDets, stateDim = opt.max_n, opt.max_m, opt.state_dim
    local nClasses = maxDets+1

    local tarFlat = nn.Reshape(1,maxTargets * stateDim,true)(inp) 	-- flatten targets in 1 x ND vector
    local detX = nn.Replicate(maxTargets,2,2)(det) 			-- replicate dets to match number of targets (M x ND)
    detX = nn.Reshape(maxDets,maxTargets*stateDim,true)(detX) 		-- reshape to proper size (M x ND)
    detX=nn.JoinTable(1,2){detX, tarFlat} 				-- attach targets to dets (M+1  x  ND)

    local labX = nn.Reshape(1,maxTargets*nClasses)(lab)	-- flatten L to 1 x MC
    labX = nn.Replicate(stateDim, 2, 2)(labX)		-- expand to D x MC to match D
    labX = nn.Reshape(stateDim, maxTargets, nClasses, true)(labX)	-- reshape to D x M x C
    labX = nn.Transpose{2,3}(labX)			-- switch D and M dimensions
    labX = nn.Reshape(stateDim*maxTargets, nClasses, true)(labX)
    labX = nn.Transpose{2,3}(labX)			-- switch D and M dimensions again

    local upd = nn.CMulTable(){labX,detX}		-- multiply element-wise
    upd = nn.Sum(1,2)(upd)				-- sum up to get dot product
    return nn.Reshape(maxTargets, stateDim, true)(upd) -- return reshaped product

      --     return nn.Reshape(nClasses, maxTargets, stateDim, true)(upd) -- return reshaped product

  end


  function doDPdebug(opt, lab, det, inp)

    local batchMode = opt.mini_batch_size>1
    --     batchMode = false
    batchMode = true

    local detX = {}
    local dpjoin = {}
    if opt.max_n == 1 then
      detX = nn.JoinTable(1,2){det, inp}:annotate{name='Cat Det+State'} -- concat det and state
      dpjoin = nn.MM(){lab, detX}:annotate{name='lab X det'}
    else
      local allDP = {}
      local selDim = 1
      if batchMode then selDim = 2 end
      for i = 1,opt.max_n do
        local oneTarget = nn.Select(selDim,i)(inp)
        oneTarget = nn.Reshape(1,opt.state_dim, batchMode)(oneTarget)

        local oneLab = nn.Select(selDim,i)(lab)
        oneLab = nn.Reshape(1,opt.max_m+1, batchMode)(oneLab)

        detX = nn.JoinTable(1,2){det, oneTarget} -- concat det and state
        detX = nn.Reshape(opt.max_m+1, opt.state_dim, batchMode)(detX)


        table.insert(allDP, nn.MM(){oneLab, detX})
      end
      dpjoin = nn.JoinTable(1,2)(allDP) -- concat det and state
    end

    return nn.Reshape(opt.max_n, opt.state_dim, batchMode)(dpjoin)

  end

  -- one input for previous location
  table.insert(inputs, nn.Identity()():annotate{
    name='X_t-1',
    description='prev. state X',
    graphAttributes = {color='red'}
  })

  --     batchMode=inputs[1]:size(1)>1


  for L = 1,n do
    -- one input for previous cell state
    if LSTMMODE then table.insert(inputs, nn.Identity()():annotate{name='c^'..(L)..'_t'}) end

    -- one input for previous hiden state
    table.insert(inputs, nn.Identity()():annotate{name='h^'..(L)..'_t'})
  end


  -- detections (t+1,... t+b)
  for B = 1,b do
    table.insert(inputs, nn.Identity()():annotate{
      name='D_t',
      description='detections',
      graphAttributes = {color='yellow'}}
    )
  end


  -- det weights
  table.insert(inputs, nn.Identity()():annotate{
    name='L_t',
    description='detections',
    graphAttributes = {color='cyan'}}
  )

  -- existance
  if einp then
    table.insert(inputs, nn.Identity()())
  end




  local x, prev_d, stateDimL, xl, exi, inputSizeL
  local outputs = {}
  local pred_state = {}

  -----------------------
  -- STATE  PREDICTION --
  -----------------------
  for L=1, n do

    -- input from previous RNN
    local prev_h = inputs[opt.nHiddenInputs*L+1] -- inputs[1] is current location
    if splitModel then prev_h = nn.Narrow(2,1,h1Size)(prev_h) end

    local prev_c = {}

    if LSTMMODE then prev_c = inputs[2*L] end

    -- determine input size for current layer
    if L == 1 then
      x = inputs[1]
      inputSizeL = opt.xSize
    else
      x = pred_state[(L-1)*opt.nHiddenInputs]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      inputSizeL = h1Size
    end

    -- Do one time tick
    local next_c, next_h = {}, {}
    if RNNMODE then pred_state = RNNTick(pred_state, opt, inputSizeL, x, prev_h, L) end
    if LSTMMODE then pred_state = LSTMTick(pred_state, opt, inputSizeL, x, prev_h, prev_c, L) end



  end -- layers


  -- set up the decoder for state prediction
  local top_pred_state = pred_state[#pred_state]
  if dropout > 0 then top_pred_state = nn.Dropout(dropout)(top_pred_state) end
  local projState = nn.Linear(h1Size, opt.xSize)(top_pred_state):annotate{
    name='state X*_t',
    description='state prediction',
    graphAttributes = {color='blue'}
  }



  -- detection
  local det_x  = {}
  --   for B = 1,b do

  local inSize = opt.state_dim
  local detInp = inputs[detInpInd]
  local LInp = nn.Exp()(inputs[labInpInd])
  --     local LInp = nn.Exp()(nn.Narrow(2,1,maxDets+1)(inputs[labInpInd+B]))

  --     batchMode=false
  local inp=nn.Reshape(opt.max_n, opt.state_dim, batchMode)(projState):annotate{name='Rshp Pred'}
  local det=nn.Reshape(opt.max_m, opt.state_dim, batchMode)(detInp):annotate{name='Rshp det'}
  local lab=nn.Reshape(opt.max_n, opt.max_m+1, batchMode)(LInp):annotate{name='Rshp lab'}

  local dp = doDP(opt, lab, det, inp)

  --     if einp then
  --       local exVec = nn.Replicate(opt.state_dim,2,0)(nn.Reshape(opt.max_n, 1)(inputs[exInpInd]))
  --       dp = nn.CMulTable(){exVec,dp}
  --       local ex = nn.Linear(maxTargets, rnnSize)(inputs[exInpInd])


  --     end
  dp = nn.Reshape(opt.xSize, batchMode)(dp)
  detInp = nn.Linear(opt.xSize,h1Size)(dp):annotate{name='Det Inp'}

  --     dp = nn.Reshape(xSize*nClasses, batchMode)(dp)
  --     detInp = nn.Linear(xSize*nClasses,h1Size)(dp):annotate{name='Det Inp'}

  table.insert(det_x, detInp)


  --     local inSize = 1
  --     detInp = nn.DotProduct(){detInp, LInp}:annotate{name='Dot(Det, Lab)'} inSize=1
  --     detInp = nn.Linear(inSize,rnnSize)(detInp):annotate{name='Det Input'}
  --     table.insert(det_x, detInp)


  --   end


  --------------------
  -- STATE  UPDATE  --
  --------------------
  -- connect top hidden output to new hidden input
  local upd_state = {}
  for L=1, n do
    -- input from previous RNN

    local prev_h = pred_state[L] -- inputs[1] is current location
    if splitModel then prev_h = nn.Narrow(2,1,h1Size)(prev_h) end


    if dropout > 0 then prev_h = nn.Dropout(dropout)(prev_h) end

    -- x is location

    -- determine input size for current layer
    if L == 1 then
      x = top_h_DA

      stateDimL = opt.xSize
    else
      x = upd_state[(L-1)+(L+1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      stateDimL = h1Size
    end

    -- only at first layer
    --     if L == 1 then
    --       for B = 1,b do	-- add detections from future
    -- 	local deth = nn.Linear(dSize,rnnSize)(det_x[B]):annotate{name='Det Input'}
    -- 	i2h = deth
    --       end
    --     end
    i2h=det_x[1]

    --     i2h = nn.CAddTable(){i2h, top_pred_state}:annotate{name='Det+Pred',
    --       graphAttributes = {color='red'}}


    -- RNN tick
    local h2h = nn.Linear(h1Size, h1Size)(prev_h):
      annotate{
        name='h2h_Upd_L'..L,
        graphAttributes = {color='red'}
      }
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h}:
      annotate{name='Det+Pred+Upd',
        graphAttributes = {color='red'}}):
      annotate{name='next_Upd_h_L'..L,
        graphAttributes = {color='red'}}

    table.insert(upd_state, next_h)
  end -- layers


  -- set up the decoder for state update
  local top_h_Upd_state = upd_state[#upd_state]
  if dropout > 0 then top_h_Upd_state = nn.Dropout(dropout)(top_h_Upd_state) end
  local projStateUpdate = nn.Linear(h1Size, opt.xSize)(top_h_Upd_state):annotate{
    name='state X_t',
    description='state update',
    graphAttributes = {color='red'}
  }



  --------------------
  --  TERMINATION   --
  --------------------
  local ter, ter2 = {}, {}
  local top_Ter_state = {}
  if einp then
    --       local exVec = nn.Replicate(opt.state_dim,2,0)(nn.Reshape(opt.max_n, 1)(inputs[exInpInd]))
    --       dp = nn.CMulTable(){exVec,dp}
    ter = nn.Linear(opt.max_n, h2Size)(inputs[exInpInd])
    if splitModel then
      local prev_h = nn.Narrow(2,h1Size+1,h2Size)(inputs[2])
      ter = nn.CAddTable(){ter, prev_h}
    end

    ter = nn.Tanh()(ter)
    if dropout > 0 then ter = nn.Dropout(dropout)(ter) end -- apply dropout, if any
    --     ter = nn.CAddTable(){top_h_Upd_state, ter}
    --     top_Ter_state = nn.CAddTable(){nn.Linear(maxTargets*nClasses, h2Size)(nn.Reshape(maxTargets*nClasses, batchMode)(LInp)), ter}
    top_Ter_state = nn.CAddTable(){nn.Linear(opt.max_n, h2Size)(nn.Select(3,opt.nClasses)(nn.Reshape(opt.max_n,opt.nClasses, batchMode)(LInp))), ter}
    top_Ter_state = nn.Reshape(h2Size, batchMode)(top_Ter_state)
    ter = nn.Linear(h2Size, opt.max_n)(top_Ter_state)
    ter = nn.Sigmoid()(ter)

    if exSmoothLoss then
      ter2 = nn.Abs()(nn.CSubTable(){ter, inputs[exInpInd]}):
        annotate{graphAttributes={color=colorTermination}}
    end

  end



  -- insert hidden states to outpu
  --   for L=1,n do table.insert(outputs, upd_state[L]) end

  local finalH = top_h_Upd_state
  if splitModel then
    finalH = nn.JoinTable(1,2){top_h_Upd_state, top_Ter_state}
    finalH = nn.Reshape(1,rnnSize,batchMode)(finalH)
  end
  table.insert(outputs, finalH)

  -- add state update (final)
  table.insert(outputs, projStateUpdate)

  -- add state prediction
  if predLoss then
    table.insert(outputs, projState)
  end

  if exLoss then
    table.insert(outputs, ter)
    if exSmoothLoss then
      table.insert(outputs, ter2)
    end
  end

  -- dynamic smoothness
--  if dynSmoothLoss then
--    table.insert(outputs, nn.Identity()(projStateUpdate))
--    --     table.insert(outputs, nn.Abs()(nn.CSubTable(){projStateUpdate, projState}))
--  end



  return nn.gModule(inputs, outputs)

end
return RNN
