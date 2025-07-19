# Load required libraries
library(circlize)
library(dplyr)
library(scales)

# # Save to transparent high-res PNG
png(filename = "multiple_light_seqs_all_combined_chord_diagram_final_mismatches_colored.png",
    width = 6000, height = 6000, res = 600, bg = "transparent")

par(family = "Arial")


# Define both confusion matrices
# heavy_to_gen_light <- matrix(c(12665, 1212, 33619, 11343), 
#                              nrow = 2, byrow = TRUE)
heavy_to_gen_light <- matrix(c(125118, 13650, 233335, 216186), 
                             nrow = 2, byrow = TRUE)
rownames(heavy_to_gen_light) <- c("Heavy_naive", "Heavy_memory")
colnames(heavy_to_gen_light) <- c("Gen_light_naive", "Gen_light_memory")

# true_to_gen_light <- matrix(c(17168, 1510, 29116, 11045), 
#                             nrow = 2, byrow = TRUE)
true_to_gen_light <- matrix(c(165079, 21697, 193374, 208139), 
                            nrow = 2, byrow = TRUE)
rownames(true_to_gen_light) <- c("True_light_naive", "True_light_memory")
colnames(true_to_gen_light) <- c("Gen_light_naive", "Gen_light_memory")

# Combined flow data
chord_data <- data.frame(
  from = c("Heavy_naive", "Heavy_naive", "Heavy_memory", "Heavy_memory",
           "True_light_naive", "True_light_naive", "True_light_memory", "True_light_memory"),
  to = c("Gen_light_naive", "Gen_light_memory", "Gen_light_naive", "Gen_light_memory",
         "Gen_light_naive", "Gen_light_memory", "Gen_light_naive", "Gen_light_memory"),
  value = c(heavy_to_gen_light[1, 1], heavy_to_gen_light[1, 2],
            heavy_to_gen_light[2, 1], heavy_to_gen_light[2, 2],
            true_to_gen_light[1, 1], true_to_gen_light[1, 2],
            true_to_gen_light[2, 1], true_to_gen_light[2, 2])
)

# Sector order - MODIFIED: HC on left, Generated LC on right
sectors <- c("Gen_light_memory", "Gen_light_naive",
             "True_light_memory", "True_light_naive", 
             "Heavy_memory", "Heavy_naive")

# Compute sector sizes
sector_sizes <- c(
  sum(heavy_to_gen_light[, 2]) + sum(true_to_gen_light[, 2]),
  sum(heavy_to_gen_light[, 1]) + sum(true_to_gen_light[, 1]),
  sum(true_to_gen_light[2, ]),
  sum(true_to_gen_light[1, ]),
  sum(heavy_to_gen_light[2, ]),
  sum(heavy_to_gen_light[1, ])
)
names(sector_sizes) <- sectors

# --- Colors ---

# Base state colors
state_colors <- c(
  "naive" = "#83aff0",  
  "memory" = "#2c456b"   
)
state_colors_extended <- c(state_colors, mixed = "#999999")

# Sector color assignment
sector_states <- ifelse(grepl("naive", sectors), "naive", "memory")
sector_colors <- state_colors[sector_states]
names(sector_colors) <- sectors

# Link color logic
# link_states <- ifelse(
#   grepl("naive", chord_data$from) & grepl("naive", chord_data$to), "naive",
#   ifelse(grepl("memory", chord_data$from) & grepl("memory", chord_data$to), "memory", "mixed")
# )
# link_colors <- ifelse(link_states == "mixed",
#                       add_transparency(state_colors_extended[link_states], 0.5),
#                       add_transparency(state_colors_extended[link_states], 0.25))

# Link color logic - MODIFIED for specific mismatch colors
link_colors <- character(nrow(chord_data))

for (i in 1:nrow(chord_data)) {
  from_sector <- chord_data$from[i]
  to_sector <- chord_data$to[i]
  
  # Special case 1: Heavy memory -> Generated light naive (mismatch)
  if (from_sector == "Heavy_memory" && to_sector == "Gen_light_naive") {
    link_colors[i] <- add_transparency("#c1440e", 0.5)
  }
  # Special case 2: Heavy naive -> Generated light memory (mismatch)
  else if (from_sector == "Heavy_naive" && to_sector == "Gen_light_memory") {
    link_colors[i] <- add_transparency("#c97b6d", 0.5)
  }
  # Special case 3: True light memory -> Generated light naive (mismatch)
  else if (from_sector == "True_light_memory" && to_sector == "Gen_light_naive") {
    link_colors[i] <- add_transparency("#c1440e", 0.5)
  }
  # Special case 4: True light naive -> Generated light memory (mismatch)
  else if (from_sector == "True_light_naive" && to_sector == "Gen_light_memory") {
    link_colors[i] <- add_transparency("#c97b6d", 0.5)
  }
  # Default logic for other cases
  else {
    from_state <- ifelse(grepl("naive", from_sector), "naive", "memory")
    to_state <- ifelse(grepl("naive", to_sector), "naive", "memory")
    
    if (from_state == to_state) {
      # Matching states
      link_colors[i] <- add_transparency(state_colors[from_state], 0.25)
    } else {
      # Other mismatches (mixed)
      link_colors[i] <- add_transparency(state_colors_extended["mixed"], 0.5)
    }
  }
}

# --- Circos plot ---

# Clear plot
circos.clear()

# Setup
circos.par(start.degree = 90, gap.degree = 3, track.margin = c(0.01, 0.01))

# Initialize with xlim
xlim_matrix <- cbind(rep(0, length(sectors)), sector_sizes)
circos.initialize(sectors = sectors, xlim = xlim_matrix)

# 1. Outer group label track
circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  sector.name <- get.cell.meta.data("sector.index")
  xlim <- get.cell.meta.data("xlim")
  
  if (grepl("Heavy", sector.name)) {
    group_label <- "HC"
    label_color <- "#36454f"
  } else if (grepl("True_light", sector.name)) {
    group_label <- "Ref LC"
    label_color <- "#6e7f80"
  } else {
    group_label <- "Generated LC"
    label_color <- "#888888"
  }
  
  circos.rect(xlim[1], 0, xlim[2], 1, col = alpha(label_color, 0.7), border = NA)
  circos.text(mean(xlim), 0.5, group_label, 
              facing = "bending", niceFacing = TRUE,
              col = "white", font = 2, cex = 2, adj = c(0.5, 0.5))
}, track.height = 0.14, bg.border = NA)

# 2. Inner memory/naive label + fill track (after!)
circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  sector.name <- get.cell.meta.data("sector.index")
  xlim <- get.cell.meta.data("xlim")
  
  # Fill with state color
  circos.rect(xlim[1], 0, xlim[2], 1, 
              col = sector_colors[sector.name], 
              border = NA, lwd = 1)
  
  # Label
  simple_name <- ifelse(grepl("memory", sector.name), "Memory", "Naive")
  circos.text(mean(xlim), 0.6, simple_name,
              facing = "bending", niceFacing = TRUE,
              col = "white", font = 2, cex = 1.9, adj = c(0.5, 0.7))
}, track.height = 0.12, bg.border = NA)

# --- Ribbon positioning ---

# Cumulative values for from sectors
from_cumsum <- data.frame()
for (sector in unique(chord_data$from)) {
  sector_data <- chord_data[chord_data$from == sector, ]
  sector_data <- sector_data[order(sector_data$to), ]
  cumsum_vals <- cumsum(c(0, sector_data$value[-nrow(sector_data)]))
  from_cumsum <- rbind(from_cumsum, data.frame(
    sector = sector, to = sector_data$to,
    start = cumsum_vals,
    end = cumsum_vals + sector_data$value
  ))
}

# Cumulative values for to sectors
to_cumsum <- data.frame()
for (sector in unique(chord_data$to)) {
  sector_data <- chord_data[chord_data$to == sector, ]
  sector_data <- sector_data[order(sector_data$from), ]
  cumsum_vals <- cumsum(c(0, sector_data$value[-nrow(sector_data)]))
  to_cumsum <- rbind(to_cumsum, data.frame(
    sector = sector, from = sector_data$from,
    start = cumsum_vals,
    end = cumsum_vals + sector_data$value
  ))
}

# Draw links
for (i in seq_len(nrow(chord_data))) {
  from_sector <- chord_data$from[i]
  to_sector <- chord_data$to[i]
  
  from_pos <- from_cumsum[from_cumsum$sector == from_sector & from_cumsum$to == to_sector, ]
  to_pos   <- to_cumsum[to_cumsum$sector == to_sector & to_cumsum$from == from_sector, ]
  
  # Draw
  if (nrow(from_pos) > 0 && nrow(to_pos) > 0) {
    circos.link(
      sector.index1 = from_sector,
      point1 = c(from_pos$start, from_pos$end),
      sector.index2 = to_sector,
      point2 = c(to_pos$start, to_pos$end),
      col = link_colors[i],
      border = NA,
      lwd = 1,
      h = 0.9
    )
  }
}

# Finalize
dev.off()