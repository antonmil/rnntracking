------------------------------------------------------
-- Online Data Association using RNNs
-- 
-- A. Milan, S. H. Rezatofighi, K. Schindler, A. Dick, I. Reid
-- arxiv 2016
--
-- @author Anton Milan (anton.milan@adelaide.edu.au)
--
-- This code is based on A. Karpathy's character-level
-- RNN https://github.com/karpathy/char-rnn
------------------------------------------------------


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'lfs'
require 'image'

require 'util.misc'  -- miscellaneous