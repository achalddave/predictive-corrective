local function save_git_info(output_dir)
    os.execute('./dump_git_info.sh ' .. output_dir)
end

local function copy_file_naive(in_path, out_path)
    -- TODO(achald): Use a library function, if one exists.
    local in_file = io.open(in_path, 'r')
    local in_contents = in_file:read('*all')
    in_file:close()
    local out_file = io.open(out_path, 'w')
    out_file:write(in_contents)
    out_file:close()
end

return {
    copy_file_naive = copy_file_naive,
    save_git_info = save_git_info
}
