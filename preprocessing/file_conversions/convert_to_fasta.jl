using CSV
using DataFrames

# Read the CSV file
#input_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/small_test_data/small_paired_data_for_analysis.csv"  
input_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_test_data_extraction_with_header.txt"
df = CSV.read(input_file, DataFrame)

# Open the FASTA file for writing
#output_file = open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/small_test_data/small_paired_data_sequence_alignment_light.fasta", "w")
output_file = open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.fasta", "w")


# Iterate over each row and write to the FASTA file
for (i, row) in enumerate(eachrow(df))
    sequence = row[:sequence_alignment_light]
    if !ismissing(sequence)
        println(output_file, ">$(i)")
        println(output_file, sequence)
    end
end

# Close the FASTA file
close(output_file)