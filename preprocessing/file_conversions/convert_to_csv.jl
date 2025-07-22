# using CSV
# using DataFrames

# println("Conversion complete! CSV written to $(output_file)")

# filename = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/pairedngs_fasta.tar"

# for line in eachline(filename)
#     # If line does not start with '>', print it
#     if !startswith(line, ">")
#         println(line)
#     end
# end

# sort -u /ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/sequences_only.txt | wc -l

using CSV
using DataFrames

function fasta_to_csv(input_fasta::String, output_csv::String)
    # We'll accumulate records in memory, then write them out at the end.
    df = DataFrame(ID=String[], Sequence=String[])

    current_id = ""
    current_seq = String[]  # store partial lines here

    for line in eachline(input_fasta)
        if startswith(line, ">")
            # If we have an ongoing record, push it to df
            if !isempty(current_id)
                push!(df, (current_id, join(current_seq, "")))
            end
            # Start a new record
            current_id = line[2:end]  # remove the leading '>'
            empty!(current_seq)       # reset sequence array
        else
            # Sequence line, accumulate
            push!(current_seq, line)
        end
    end

    # After the loop, if there's a final record waiting, push it
    if !isempty(current_id)
        push!(df, (current_id, join(current_seq, "")))
    end

    # Write to CSV
    CSV.write(output_csv, df)
    println("Wrote CSV to $output_csv")
end

# Example usage:
fasta_to_csv("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/pairedngs_fasta.tar", "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/pairedngs_fasta.csv")