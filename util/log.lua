--
-- log.lua
--
-- Copyright (c) 2016 rxi
--
-- This library is free software; you can redistribute it and/or modify it
-- under the terms of the MIT license. See LICENSE for details.
--

local log = { _version = "0.1.0" }

log.usecolor = true
log.outfile = nil
log.level = "trace"

local modes = {
    { name = "trace", color = "\27[34m", },
    { name = "debug", color = "\27[36m", },
    { name = "info",  color = "\27[32m", },
    { name = "warn",  color = "\27[33m", },
    { name = "error", color = "\27[31m", },
    { name = "fatal", color = "\27[35m", },
}

local levels = {}
for i, v in ipairs(modes) do
    levels[v.name] = i
end

local _tostring = tostring

local tostring = function(...)
    local t = {}
    for i = 1, select('#', ...) do
        local x = select(i, ...)
        t[#t + 1] = _tostring(x)
    end
    return table.concat(t, " ")
end

-- Taken from Torch REPL's init.lua.
local ndepth = 4
function print_no_color(...)
   local function rawprint(o)
      io.write(tostring(o or '') .. '\n')
      io.flush()
   end
   local objs = {...}
   local function printrecursive(obj,depth)
      local depth = depth or 0
      local tab = depth*4
      local line = function(s) for i=1,tab do io.write(' ') end rawprint(s) end
      if next(obj) then
         line('{')
         tab = tab+2
         for k,v in pairs(obj) do
            if type(v) == 'table' then
               if depth >= (ndepth-1) or next(v) == nil then
                  line(tostring(k) .. ' : {...}')
               else
                  line(tostring(k) .. ' : ') printrecursive(v,depth+1)
               end
            else
               line(tostring(k) .. ' : ' .. v)
            end
         end
         tab = tab-2
         line('}')
      else
         line('{}')
      end
   end
   for i = 1,select('#',...) do
      local obj = select(i,...)
      if type(obj) ~= 'table' then
         if type(obj) == 'userdata' or type(obj) == 'cdata' then
            rawprint(obj)
         else
            io.write(obj .. '\t')
            if i == select('#',...) then
               rawprint()
            end
         end
      elseif getmetatable(obj) and getmetatable(obj).__tostring then
         rawprint(obj)
      else
         printrecursive(obj)
      end
   end
end
local torch_mode = print_new ~= nil and print == print_new
for i, x in ipairs(modes) do
    local nameupper = x.name:upper()
    log[x.name] = function(...)
        -- Return early if we're below the log level
        if i < levels[log.level] then
            return
        end

        local msg = tostring(...)
        -- In case you want to include line info.
        -- local info = debug.getinfo(2, "Sl")
        -- local lineinfo = info.short_src .. ":" .. info.currentline

        -- Output to console. The console version is colored.
        io.write(string.format("%s[%-6s%s]%s: ",
                            log.usecolor and x.color or "",
                            nameupper,
                            os.date('%X'),
                            log.usecolor and "\27[0m" or ""))
        -- If in torch mode, we can use print for both stdout and for file,
        -- since it uses io.write under the hood, which can be redericted to a
        -- file. Otherwise, use io.write to ensure consistency between stdout
        -- and file, since we can't redirect the usual print.
        if torch_mode then
            print(...)
        else
            io.write(msg, '\n')
        end

        -- Output to log file
        if log.outfile then
            local fp = io.open(log.outfile, "a")
            fp:write(string.format("[%-6s%s]: ", nameupper, os.date('%X')))
            if torch_mode then
                io.output(fp)
                print_no_color(...)
                io.output(io.stdout)
            else
                fp:write(msg .. '\n')
            end
            fp:close()
        end
    end
end

return log
