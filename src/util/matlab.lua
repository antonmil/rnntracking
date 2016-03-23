--- A set of tools similar to Matlab
-- Forked from TODO?

--------------------------------------------
-- SET OPERATIONS ON TABLES
--------------------------------------------   

set={}

function set.new(t)
   local set={}
   for _,l in ipairs(t) do set[l]=true end
   return set
end

function set.union(a,b)
   local res=set.new{}
   for k in pairs(a) do res[k]=true end
   for k in pairs(b) do res[k]=true end
   return res
end

function set.intersect(a,b)
   local res=set.new{}
   for k in pairs(a) do res[k]=b[k] end
   return res
end

function set.difference(a,b)
   local res=set.new{}
   for k in pairs(a) do
      if not b[k] then res[k]=true end
   end
return res
end

function set.table(s)
   local t={}
   for k in pairs(s) do t[#t+1]=k end
   return t
end

function any(x)
   return torch.sum(x:ne(0))>0
end

-- convert the 1D tensor x to a table
function torch.Table(x)
   if x ~= nil then
      local t={}
      for i=1,(#x)[1] do
         table.insert(t,x[i])
      end
      return t
   else
      return {}
   end
end


---------------------------------------------------------
-- some MATLAB-like set operations on 1D tensors
---------------------------------------------------------

--------------------------------------------------------------------------
--- Set difference of two tensors (result is sorted).
function torch.setdiff(a,b)
   local res=set.table(set.difference(set.new(torch.Table(a)),set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

--------------------------------------------------------------------------
--- Set union of two tensors.
function torch.union(a,b)
   local res=set.table(set.union(set.new(torch.Table(a)),set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

--------------------------------------------------------------------------
--- Set intersection of two tensors.
function torch.intersect(a,b)
   local res=set.table(set.intersect(set.new(torch.Table(a)),set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

--------------------------------------------------------------------------
--- Rreturns the indices of non-zero elements of a 1D tensor.
function torch.find(x)
  if x:nDimension() > 1 then
    error('torch.find is only defined for 1D tensors')
  end
  local indx={}
  for i=1,(#x)[1] do
    if x[i]>0 then table.insert(indx,i) end
  end
  return torch.IntTensor(indx) -- convert table back to tensor
end