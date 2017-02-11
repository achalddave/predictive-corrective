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
        print(string.format("%s[%-6s%s]%s: %s",
                            log.usecolor and x.color or "",
                            nameupper,
                            os.date('%X'),
                            log.usecolor and "\27[0m" or "",
                            msg))

        -- Output to log file
        if log.outfile then
            local fp = io.open(log.outfile, "a")
            local str = string.format("[%-6s%s]: %s\n",
                                      nameupper, os.date('%X'), msg)
            fp:write(str)
            fp:close()
        end
    end
end

return log
