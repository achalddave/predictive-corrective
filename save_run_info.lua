function save_git_info(output_dir)
    os.execute('./dump_git_info.sh ' .. output_dir)
end

return {
    save_git_info = save_git_info
}
