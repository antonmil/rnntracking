--- Visualize boxes on a sequence

require 'torch'
require 'gnuplot'
require 'image'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Display boxes on a video')
cmd:text()
cmd:text('Options')
-- model path
cmd:option('-seqName','TUD-Campus','Sequence Name')
-- options
cmd:option('-file','gt','gt, det, or result-file-name')
cmd:option('-length',0,'how many frames to show (0=full sequence)')

cmd:text()
-- parse input params
visopt = cmd:parse(arg)

local seqName = visopt.seqName
local imgHeight, imgWidth = getImgRes(seqName)
imgWidth = 1 -- no need for scaling

if string.lower(visopt.file) == "res" then
  visopt.file = string.format("out/%s.txt",seqName)
end

if string.lower(visopt.file) == "gt" then
  tracks = getGTTracks(seqName) * imgWidth
elseif string.lower(visopt.file) == "det" then
  error("Det vis not implemented")
else
  res = readTXT(visopt.file, 1)
  F = tabLen(res) -- how many frames
  print(F..' frames read')
  if visopt.length~=0 and visopt.length<F then
    F=visopt.length
    print('Trimmed to '..F..' frames');
  end
  
  -- find maxID
  local minN, maxN = 1000,-1000
  for t=1,F do
    for k in pairs(res[t]) do 
      if k<minN then minN=k end
      if k>maxN then maxN=k end
    end
  end
  print(string.format("min ID: %d. max ID: %d\n",minN, maxN))
  
  local nDim=4
  

  -- allocate tracks tensor
  local N = maxN		-- number of tracks
  tracks = torch.ones(N, F, nDim):fill(0)
  
  -- Now create a FxN tensor with states
  for t=1,F do
    for k,v in pairs(res[t]) do
      tracks[k][t]=v
    end
  end  
end

local N,F = getDataSize(tracks)


------------------------------------------------
-------- Scene Info ----------------------------
-- local seqName = 'TUD-Campus'
-- local seqName = 'TUD-Crossing'
-- local seqName = 'PETS09-S2L1'
-- local seqName = 'ADL-Rundle-1'
-- local seqName = 'KITTI-16'

imFile = getDataDir() .. seqName .. "/img1/000001.jpg"
-- im = image.load(imFile)
-- image.display(im)
-- gnuplot.raw("set size ratio -1")
imPlot = string.format("plot '%s' binary filetype=jpg with rgbimage",imFile)
gnuplot.raw(imPlot)
-- gnuplot.raw("set object 1 rect from 23,34 to 342,38 fillstyle empty fc lt 2 front")
gnuplot.plotflush()


------------------------------------------------------------------
--- Flip box vertically for gnuplot
-- @param box		A bounding box with cx,cy,w,h coordinates
-- @param imgHeight	The image height to flip vertically
-- @return A box for rect format: [x1,y1,x2,y2]
function getGnuplotBox(box, imgHeight)
  gnuBox = box:clone()
  gnuBox[1] = box[1] - box[3]/2		-- shift xcenter to left
  gnuBox[2] = imgHeight - (box[2]-box[4]/2) -- flip y1 after ycenter to top
  gnuBox[3] = gnuBox[1] + box[3]	-- x2
  gnuBox[4] = gnuBox[2] - box[4]	-- y2
  return gnuBox
end

-- now run through sequence
for t=1,F do
    winIDstr = string.format("set term wxt %d reset",1)
    gnuplot.raw(winIDstr)
  imFile = getDataDir() .. seqName .. string.format("/img1/%06d.jpg",t)
  imPlot = string.format("plot '%s' binary filetype=jpg with rgbimage",imFile)
--   if t<5 then
  gnuplot.raw(imPlot)  
  gnuplot.raw('unset key; unset tics;')
--   end
  for i=1,N do
--     print(tracks[i][t])
    box = getGnuplotBox(tracks[i][t], imgHeight)
--     if box[1]>0 then    
    
      bxStr = string.format("set object %d rect from %d,%d to %d,%d front lw 3 fs empty  border lc rgb %s", i,box[1],box[2],box[3],box[4],getColorFromID(i))
--       bxStr = string.format("set object %d arrow from %d,%d to %d,%d lw 3 front lc rgb \'red\' nohead", i,box[1],box[2],box[3],box[4])
      gnuplot.raw(bxStr)    
--     else
--       bxStr = string.format("unset set object %d", i)
--     end
    
  end  
--   print(t)
  sleep(.1) -- slow down and prevent crashing hack
  gnuplot.plotflush()
end
