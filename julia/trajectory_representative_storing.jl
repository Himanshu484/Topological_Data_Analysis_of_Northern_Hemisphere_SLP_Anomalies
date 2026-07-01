using NPZ
using Plots
using Ripserer
using JSON

notebook_dir = pwd()

# Read files for specific years with data type option (from your earlier function)
function read_slp_files_by_years(years, data_type="sub", base_folder=notebook_dir*"/data/processed_data/SLP_data_years")
    slp_data = Dict()  # Dictionary to store data by year
    
    # Validate data_type
    if !(data_type in ["sup", "sub"])
        error("data_type must be either 'sup' or 'sub', got: $data_type")
    end
    
    for year in years
        year_folder = joinpath(base_folder, string(year))
        
        if !isdir(year_folder)
            continue
        end
        
        # Get all .npy files for this year and data type
        files = filter(f -> endswith(f, ".npy") && startswith(f, "slp_$(data_type)_$(year)_day_"), 
                      readdir(year_folder))
        sort!(files)  # Sort by filename
        
        year_data = []
        filenames = []
        
        for file in files
            filepath = joinpath(year_folder, file)
            data = npzread(filepath)
            push!(year_data, data)
            push!(filenames, file)
        end
        
        slp_data[year] = Dict("data" => year_data, "filenames" => filenames)
    end
    
    return slp_data
end

# Convert CartesianIndex to numerical tuples (0-based for Python)
function convert_to_numerical(all_list_data)
    numerical_data = []
    for list_item in all_list_data
        converted_item = []
        for tuple_pair in list_item
            start_idx = tuple_pair[1]
            end_idx = tuple_pair[2]
            
            # Convert to 0-based indexing
            start_coords = [start_idx[1] - 1, start_idx[2] - 1]
            end_coords = [end_idx[1] - 1, end_idx[2] - 1]
            
            push!(converted_item, [start_coords, end_coords])
        end
        push!(numerical_data, converted_item)
    end
    return numerical_data
end

# Convert death simplex vertices to 0-based coordinates
function convert_death_simplex_vertices(all_list_data_death)
    converted_death_simplices = []
    for death_simplex in all_list_data_death
        # death_simplex is a 4-tuple of CartesianIndex{2}
        converted_vertices = []
        for vertex in death_simplex
            # Convert each CartesianIndex to 0-based coordinates
            converted_vertex = [vertex[1] - 1, vertex[2] - 1]
            push!(converted_vertices, converted_vertex)
        end
        push!(converted_death_simplices, converted_vertices)
    end
    return converted_death_simplices
end

# Process representative data for multiple years
function process_representatives_by_years(years, data_type="sub", 
                                        base_input_folder=notebook_dir*"/data/processed_data/SLP_data_years",
                                        base_output_folder=notebook_dir*"/data/processed_data/representative_data")
    
    println("Processing $(data_type) data for years: $years")
    
    # Load all data for specified years
    all_data = read_slp_files_by_years(years, data_type, base_input_folder)
    
    # Process each year
    for (year_idx, year) in enumerate(years)
        if !(year in keys(all_data))
            println("Year $year: No data found - SKIPPED")
            continue
        end
        
        # Create output directory for this year
        year_output_dir = joinpath(base_output_folder, string(year))
        mkpath(year_output_dir)
        
        year_data = all_data[year]["data"]
        year_filenames = all_data[year]["filenames"]
        total_files = length(year_data)
        
        print("Year $year ($(year_idx)/$(length(years))): ")
        
        # Process each file in this year
        for (i, (data, filename)) in enumerate(zip(year_data, year_filenames))
            
            # Compute persistence
            input_data = data
            result = ripserer(Cubical(input_data), reps=true, alg=:homology)
            
            # Extract and convert data
            births = [item.birth for item in result[2]]
            deaths = [item.death for item in result[2]]
            all_list_data = [vertices.(item.representative) for item in result[2]]
            numerical_list_data = convert_to_numerical(all_list_data)
            
            # Extract death simplex vertices and convert to 0-based coordinates
            all_list_data_death = [vertices(item.death_simplex) for item in result[2]]
            converted_death_simplices = convert_death_simplex_vertices(all_list_data_death)
            
            # Create data dictionary
            data_dict = Dict(
                "year" => year,
                "data_type" => data_type,
                "births" => births,
                "deaths" => deaths,
                "list_data" => numerical_list_data,
                "death_simplex_vertices" => converted_death_simplices,
                "original_filename" => filename
            )
            
            # Create output filename (replace .npy with .json)
            output_filename = replace(filename, ".npy" => ".json")
            output_path = joinpath(year_output_dir, output_filename)
            
            # Save as JSON
            open(output_path, "w") do f
                JSON.print(f, data_dict)
            end
            
            # Show percentage progress
            percentage = round((i / total_files) * 100, digits=1)
            print("\rYear $year ($(year_idx)/$(length(years))): $(percentage)% ($(i)/$(total_files))")
        end
        
        println(" - COMPLETED ✓")
    end
    
    println("All processing complete! Files saved to: $base_output_folder")
end

# Usage examples:
# Process 5 years of sublevel data
years_to_process = collect(1948:2024)  # 1948, 1949, 1950, 1951, 1952

println("="^60)
println("PROCESSING SUBLEVEL DATA")
println("="^60)
process_representatives_by_years(years_to_process, "sub")

println("\n" * "="^60)
println("PROCESSING SUPERLEVEL DATA") 
println("="^60)
process_representatives_by_years(years_to_process, "sup")

println("\n" * "="^60)
println("ALL PROCESSING COMPLETED!")
println("="^60)