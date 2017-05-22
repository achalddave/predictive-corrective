local log = require 'util/log'

local function save_git_info(output_dir)
    os.execute('./dump_git_info.sh ' .. output_dir)
end

local function copy_file(in_path, out_path, preserve)
    --[[
    -- Args:
    --    in_path (str)
    --    out_path (str)
    --    preserve (bool): If true, pass "--preserve=all" flag to cp.
    --
    -- TODO(achald): See if there is a portable way to copy a file in lua.
    ]]--
    preserve = preserve == nil and false
    local flags = preserve == true and '-v --preserve=all' or '-v'
    local cmd = string.format('cp %s %s %s', flags, in_path, out_path)
    log.info('Executing command:', cmd)
    os.execute(cmd)
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
    copy_file = copy_file,
    save_git_info = save_git_info,
    read_and_increment_experiment_id = read_and_increment_experiment_id
}
