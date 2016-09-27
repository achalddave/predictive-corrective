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

local function read_and_increment_experiment_id(experiment_id_path)
    local experiment_id_file = io.open(experiment_id_path, 'r')
    local experiment_id = experiment_id_file:read('*number')
    experiment_id_file:close()

    experiment_id_file = io.open(experiment_id_path, 'w')
    experiment_id_file:write(experiment_id + 1)
    experiment_id_file:close()

    return experiment_id
end

return {
    copy_file_naive = copy_file_naive,
    save_git_info = save_git_info,
    read_and_increment_experiment_id = read_and_increment_experiment_id
}
