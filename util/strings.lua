local function trim(str)
    return str:gsub("^%s*(.-)%s*$", "%1")
end

local function starts(str, start_str)
    return string.sub(str, 1, string.len(start_str)) == start_str
end

return {
    trim = trim,
    starts = starts
}
