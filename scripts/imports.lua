--[[ Useful for starting an interpreter with all the necessary imports:
--
-- Usage:
-- ~ th -i scripts/imports.lua
--
-- NOTE: Do not 'require' this file except from an interactive session. If you
-- need these imports, copy and paste them so they are explicitly imported.
--]]
local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'layers/init'
require 'cunn'
