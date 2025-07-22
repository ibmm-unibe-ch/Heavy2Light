using DataStructures

filename = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/pairedngs_fasta.tar"

unique_pairs = Set{String}()

open(filename, "r") do f
    for line in eachline(f)
        # Only process header lines (start with '>')
        if startswith(line, ">")
            parts = split(line, '|')
            # Combine the last two fields with '|'
            last_two = parts[end-1] * "|" * parts[end]
            push!(unique_pairs, last_two)
        end
    end
end

# Print the total number of unique pairs
println(length(unique_pairs))