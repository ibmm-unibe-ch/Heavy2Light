#!/home/leab/.juliaup/bin/julia

# # Extract IDs from query file
# function get_ids(file)
#     ## Populate set with lines from the file
#     queries = Set(eachline(file))
#     return queries
# end

# # Modified compare_ids function to accept a file handle for output -> IDs here: heavy[SEP]light sequences!
# function compare_ids(queries, db, output_file)
#     # Loop through db_lines
#     for ln in eachline(db)
#         # Extract ID with regex
#         parts = split(ln, ',')  # Splitting based on comma
#         #ID = strip(parts[end])  # Assuming the ID is the last part after the comma!
#         ID = strip(parts[15])  # Extracting ID from the xth column!! -> always check the column number in your dataset!
#         # Search in queries
#         if ID in queries
#             println(output_file, ln)  # Writing to file instead of standard output
#         end
#     end
# end

# function get_id(line::String)
#     # We know the line should look like:
#     # "some,stuff,with,commas",some-other-column
#     # so let's find the matching closing quote and split there.
    
#     # 1) Strip whitespace/newlines
#     line = strip(line)
    
#     # 2) Make sure it starts with a quote
#     if !startswith(line, "\"")
#         error("Line does not start with a quote, can't parse: $line")
#     end

#     # 3) Find the next quote that actually ends the field
#     #    (assuming no double quotes in the field).
#     pos = findnext(x -> x == '"', line, 2)  # start searching from index 2
    
#     # If we find a matching quote:
#     if pos === nothing
#         error("No closing quote found in line: $line")
#     end

#     # The ID is everything between the first and that closing quote
#     the_id = line[2:pos-1]
#     return the_id
# end

# function compare_ids(queries, db, output_file)
#     for ln in eachline(db)
#         id_in_line = get_id(ln)
#         if id_in_line in queries
#             println(output_file, ln)
#         end
#     end
# end


# # Open output file for writing
# open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/matched_sequences_sep-2.txt", "w") do output_file
#     # Run the actual functions with output redirection
#     open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/sequences_only_sep.txt") do query_file
#         queries = get_ids(query_file)
#         open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/coherence_analysis_in_oas_db/data/full_extraction_for_coherence_paired_data_header.csv") do db_file
#             compare_ids(queries, db_file, output_file)
#         end
#     end
# end

# # Open output file for writing
# open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/test_function_output.txt", "w") do output_file
#     # Run the actual functions with output redirection
#     open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/test_function.txt") do query_file
#         queries = get_ids(query_file)
#         open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/db_test_function.txt") do db_file
#             compare_ids(queries, db_file, output_file)
#         end
#     end
# end


# # Open output file for writing
# open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/matched_seqs_after_cdrh3_100_clust.txt", "w") do output_file
#     # Run the actual functions with output redirection
#     open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/curated_pairedNGS_seqs_cdrh3_only_100_clu_rep_only_ids_strings.csv") do query_file
#         queries = get_id(query_file)
#         open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/unique_values_filtered_pairedngs.csv") do db_file
#             compare_ids(queries, db_file, output_file)
#         end
#     end
# end


function get_id(line::String)
    line = strip(line)

    if !startswith(line, "\"")
        error("Line does not start with a quote, cannot parse: $line")
    end

    # Find the matching closing quote
    pos = findnext(x -> x == '"', line, 2)
    if pos === nothing
        error("No closing quote found in line: $line")
    end

    # Extract the text inside the quotes
    return line[2:pos-1]
end

function get_ids(file::IO; skip_header=false)
    result = Set{String}()

    if skip_header
        # Throw away the first line (header)
        readline(file; keep=false)  # or just `readline(file)` is enough
    end

    for ln in eachline(file)
        push!(result, get_id(ln))
    end
    return result
end

function compare_ids(queries::Set{String}, db::IO; skip_header=false)
    # Optionally skip db header if it exists
    if skip_header
        readline(db; keep=false)
    end

    for ln in eachline(db)
        db_id = get_id(ln)
        if db_id in queries
            println(ln)
        end
    end
end


open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/matched_seqs_after_cdrh3_100_clust-2.csv", "w") do output_file
    open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/curated_pairedNGS_seqs_cdrh3_only_100_clu_rep_only_ids_strings.csv") do query_file
        # The queries file has a header, so skip the first line:
        queries = get_ids(query_file, skip_header=true)

        open("/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/data/unique_values_filtered_pairedngs.csv") do db_file
            # If your DB file also has a header line, skip it:
            for ln in eachline(db_file)
                db_id = get_id(ln)
                if db_id in queries
                    println(output_file, ln)
                end
            end
        end
    end
end

