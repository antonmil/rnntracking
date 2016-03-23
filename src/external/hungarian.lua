function hungarian(icostMatrix)
  local max = #icostMatrix	-- rows
  local maxo = max 		-- keep original num of rows
  local maxj = #icostMatrix[1]	-- columns
  local done = false
  
--   print(max,maxj)
--   print(icostMatrix)
  if max ~= maxj then
    -- Non square matrix.  Pad 
    if max < maxj then
--       print("too many tasks, not enough workers, adding " .. maxj - max .. " 'empty' worker slots")
      for i = max+1,maxj do
	icostMatrix[i] = {}
	for j = 1,maxj do
	  icostMatrix[i][j] = 0 -- pad matrix with zeros
	end
      end
      max = maxj
--       print("Cost matrix dimensions: " ..  #icostMatrix .. "," .. #icostMatrix[1])
    else
--       return
      for i=1,max do
	for j=1,max-maxj do
	  table.insert(icostMatrix[i],0)
	end
      end
    end
  end
--   print(icostMatrix)
  
  hungarianStep1(icostMatrix,max)
  
  -- Define and initialise variables to be used from now on
  local rowCovered = {}
  local columnCovered = {}
  local starred = {}
  
  for i = 1,max do
    rowCovered[i] = false
    columnCovered[i] = false
    
    starred[i] = {}
    for j = 1,max do
      starred[i][j] = 0
    end
  end
  
  hungarianStep2(icostMatrix,max,rowCovered,columnCovered,starred)
  
  hungarianClearCovers(rowCovered,columnCovered,max) -- function to uncover all rows and columns
  
  local step = 3
  local r,c
  local iterations = 0
  
  local log = "3," -- log keeps a track of the order in which steps were called
  
  while step ~= 7 do
    if step == 3 then
      step = hungarianStep3(icostMatrix,max,rowCovered,columnCovered,starred)
    else
      if step == 4 then
	step,r,c = hungarianStep4(icostMatrix,max,rowCovered,columnCovered,starred)
      else
	if step == 5 then
	  step = hungarianStep5(icostMatrix,max,rowCovered,columnCovered,starred,r,c)
	else
	  if step == 6 then
	    step = hungarianStep6(icostMatrix,max,rowCovered,columnCovered,starred)
	  end
	end
      end
    end
    
    log = log .. step .. ","
    
    iterations = iterations + 1
    
    if step == 7 then break end
    
    if iterations > 300 then
      print("ERROR possible infinite loop in hungarian")
      break
    end
  end
  
  
--   print("hungarian finished in ".. iterations .. " iterations")
--   print("log: " .. log )
  
  
  
  --[[ DEBUG text
  print("Assignement matrix =")
  local line = "   "
  for i = 1,#starred do
  line = ""
  for j = 1,#starred[i] do
  line = line .. starred[i][j] .. " "
end
  print(line)
end--]]
  
  local assigned = {}
--   print(max)
  for i = 1,max do
    for j = 1,max do
--       print(i,j)
      if starred[i][j] == 1 and j<=maxj and i <= maxo then
	assigned[i]=j
	break
      end
    end
  end
  
--   print(assigned)
  return assigned
end
  -------------------------------------------------------
  
  
-- STEP 1 --
function hungarianStep1(icostMatrix,max)
  local lowestCost
  for i = 1,max do
    lowestCost = icostMatrix[i][1]
    for j = 2,max do
      if icostMatrix[i][j] < lowestCost then
	lowestCost = icostMatrix[i][j]
      end
    end
    
    for j = 1,max do
      icostMatrix[i][j] = icostMatrix[i][j] - lowestCost
    end
  end
end
-- END of STEP 1 --

-- STEP 2 --
function hungarianStep2(icostMatrix,max,rowCovered,columnCovered,starred)
  for i = 1,max do
    for j = 1,max do
      if icostMatrix[i][j] == 0 and rowCovered[i] == false and columnCovered[j] == false then
	starred[i][j] = 1
	rowCovered[i] = true
	columnCovered[j] = true
      end
    end
  end
end
-- END of STEP 2 --

-- STEP 3 --
function hungarianStep3(icostMatrix,max,rowCovered,columnCovered,starred)
  local count = 0
  
  for i = 1,max do
    for j = 1,max do
      if starred[i][j] == 1 then
	columnCovered[j] = true
      end
    end
  end
  
  for j = 1,max do
    if columnCovered[j] then
      count = count + 1
    end
  end
  
  if count >= max then
--     print("Hungarian solution found!")
    return 7 -- done!
  else
    return 4 -- do step 4 next
  end
end
-- END of STEP 3 --

-- STEP 4 --
function hungarianStep4(icostMatrix,max,rowCovered,columnCovered,starred)
  local done4 = false
  local step, check, scol
  local row,col
  
  local debugCount = 0
  local log = "Log:"
  
  while done4 == false do
    debugCount = debugCount + 1
    if debugCount > 300 then
      print("ERROR Infinite loop found in hungarianStep4")
      return 7
    end
    
    row,col = hungarianFindZero(icostMatrix,rowCovered,columnCovered,max)
    
    ---[[
    if row == 0 then
      done4 = true
      if log ~="Log:" then
-- 	print(log)
      end
      step = 6 -- Do step 6 next
    else
      starred[row][col] = 2
      log = log .." z="..row..","..col
      check = hungarianIsStarInRow(row,starred,max)
      if check then
	col = hungarianStarInRow(row,starred,max)
	log = log .." s="..row..","..col
	rowCovered[row] = true
	columnCovered[col] = false
      else
	done4 = true
	if log ~="AI:" then
-- 	  print(log)
	end
	step = 5 -- Do step 5 next
      end
    end
  end
  return step,row,col
end
-- END of STEP 4 --

-- STEP 5 --
function hungarianStep5(icostMatrix,max,rowCovered,columnCovered,starred,ZO_r,ZO_c)
  local r,c
  local count = 1
  local path = {}
  path[count] = {}
  path[count][1] = ZO_r
  path[count][2] = ZO_c
  local done5 = false
  
  local debugCount = 0
  
  while done5 == false do
    
    debugCount = debugCount + 1
    if debugCount > 300 then
      print("ERROR: possible infinite loop found in hungarianStep5")
      return 7 -- note, there have been occassions where this get out has been used.
      --I don't know if that's a hint as to what the problem could be, or just a symptom of the problem
    end
    
    r = hungarianStarInColumn(path[count][2],starred,max)
    if r > 0 then
      count = count + 1
      if path[count] == nil then
	path[count] = {}
      else
	print("ERROR, path[count] already exists")
      end
      path[count][1] = r
      path[count][2] = path[count-1][2]
    else
      done5 = true
    end
    if done5 == false then
      c = hungarianDoubleStarInRow(path[count][1],starred,max)
      count = count + 1
      if path[count] == nil then
	path[count] = {}
      else
	print("ERROR, path[count] already exists")
      end
      path[count][1] = path[count-1][1]
      path[count][2] = c
    end
  end
  
  -- convert path
  for i = 1,count do
    if starred [path[i][1]] [path[i][2]] == 1 then
      starred [path[i][1]] [path[i][2]] = 0
    else
      starred [path[i][1]] [path[i][2]] = 1
    end
  end
  
  rowCovered,columnCovered = hungarianClearCovers(rowCovered,columnCovered,max)
  
  -- erase double starred
  for i = 1,max do
    for j = 1,max do
      if starred[i][j] == 2 then
	starred[i][j] = 0
      end
    end
  end
  
  return 3 -- Do step 3 next
  
end
-- END of STEP 5 --

-- STEP 6 --
function hungarianStep6(icostMatrix,max,rowCovered,columnCovered,starred)
  local lowestValue = 9876543216433
  for i = 1,max do
    for j = 1,max do
      if rowCovered[i] ~= true and columnCovered[j] ~= true then
	--print("This value is uncovered")  -- This never gets printed when the program gets stuck in the loop
	if lowestValue > icostMatrix[i][j] then
	  lowestValue = icostMatrix[i][j]
	end
      end
    end
  end
  
  --if lowestValue == 9876543216433 then
  --  -- ERROR, no lowest value found
  --end
  
  --print("Lowest Value = " .. lowestValue)
  
  for i = 1,max do
    for j = 1,max do
      if rowCovered[i] then
	icostMatrix[i][j] = icostMatrix[i][j] + lowestValue
      end
      if columnCovered[j] == false then
	icostMatrix[i][j] = icostMatrix[i][j] - lowestValue
      end
    end
  end
  
  return 4 -- Do step 4 next
end
-- END of STEP 6 --

-- hungarianFindZero(row,col) --
function hungarianFindZero(icostMatrix,rowCovered,columnCovered,max)
  for i = 1,max do
    for j = 1,max do
      if icostMatrix[i][j] == 0 and rowCovered[i] == false and columnCovered[j] == false then
	return i,j
      end
    end
  end
  return 0,0
end

function hungarianIsStarInRow(row,starred,max)
  for j = 1,max do
    if starred[row][j] == 1 then
      return true
    end
  end
  return false
end     

function hungarianStarInRow(row,starred,max)
  for j = 1,max do
    if starred[row][j] == 1 then
      return j
    end
  end
  return false
end

function hungarianStarInColumn(col,starred,max)
  for i = 1,max do
    if starred[i][col] == 1 then
      return i
    end
  end
  return 0
end

function hungarianDoubleStarInRow(row,starred,max)
  for j = 1,max do
    if starred[row][j] == 2 then
      return j
    end
  end
  return 0
end

function hungarianClearCovers(rowCovered,columnCovered,max)
  for i = 1,max do
    rowCovered[i] = false
    columnCovered[i] = false
  end
  return rowCovered,columnCovered
end