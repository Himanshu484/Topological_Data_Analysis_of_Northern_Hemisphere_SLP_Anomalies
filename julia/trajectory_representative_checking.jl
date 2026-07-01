using NPZ
using Plots
using Ripserer

notebook_dir = pwd()
# Read a single file (your original example)
data = npzread(notebook_dir*"/data/processed_data/SLP_data_years/1948/slp_sub_1948_day_1.npy")

data

result = ripserer(Cubical(data), reps=true, alg=:homology)

result
result[2]

m = 60

(result[2][m].birth,result[2][m].death)

result[2][m]

chain_data = result[2][m].representative
list_data = vertices.(chain_data)
vertices(result[2][m].death_simplex)

representative_list = []
for element in list_data
    push!(representative_list,element[1])
    push!(representative_list,element[2])
end

# Create the heatmap
p = heatmap(data, color=:viridis, aspect_ratio=:equal)

# Extract coordinates from CartesianIndex and overlay as scatter points
x_coords = [idx[2] for idx in representative_list]  # Column indices (x-axis)
y_coords = [idx[1] for idx in representative_list]  # Row indices (y-axis)

# Add highlighted points
scatter!(p, x_coords, y_coords, 
         color=:red, 
         markersize=2, 
         markershape=:circle,
         markerstrokewidth=2,
         markerstrokecolor=:white,
         label="Highlighted indices")
display(p)