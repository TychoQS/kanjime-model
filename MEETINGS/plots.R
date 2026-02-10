# Script de R base para visualizar los resultados de los experimentos de entrenamiento
# Este script lee 'training_log.csv' y genera gráficos de rendimiento.

# 1. Cargar los datos
# Intentamos cargar el archivo desde la raíz si estamos en MEETINGS/ o desde el directorio actual
csv_path <- "../training_log.csv"
if (!file.exists(csv_path)) {
  csv_path <- "training_log.csv"
}

if (!file.exists(csv_path)) {
  stop("Error: No se encuentra 'training_log.csv'. Asegúrate de ejecutar el script desde la raíz o la carpeta MEETINGS.")
}

data <- read.csv(csv_path, stringsAsFactors = FALSE, na.strings = "N/A")

# 2. Limpieza de datos
# Convertir columnas de porcentaje a numéricas (quitando el símbolo %)
clean_pct <- function(x) {
  if (is.numeric(x)) return(x)
  return(as.numeric(gsub("%", "", x)))
}

data$val_accuracy  <- clean_pct(data$val_accuracy)
data$test_accuracy <- clean_pct(data$test_accuracy)
data$top_5_test    <- clean_pct(data$top_5_test)
data$top_5_casia   <- clean_pct(data$top_5_casia)
data$val_loss      <- as.numeric(data$val_loss)

# Filtrar filas vacías
data <- data[!is.na(data$experiment_id) & data$experiment_id != "", ]

# Filtrar experimentos fallidos específicos que ensucian las medias (Outliers)
data <- data[data$experiment_id != "mobilenet_v3-model-v2", ]

# 3. Generación de gráficos (Guardar en un PDF)
output_file <- "performance_plots.pdf"
pdf(output_file, width = 11, height = 8.5)

# Función auxiliar para colores elegantes
colors_main <- c("#2E5A88", "#D97D3A", "#4E9F3D", "#C84C4C", "#7B4EA3", "#A34E7B")

# Función para extraer la "línea" del experimento (p.ej., 'ghostnet-model' de 'ghostnet-model-v4')
get_experiment_line <- function(id) {
  # Eliminamos el sufijo de versión "-v1", "-v2", etc.
  line <- sub("-v[0-9]+$", "", id)
  return(line)
}

data$experiment_line <- sapply(data$experiment_id, get_experiment_line)

# --- Página 1: Val vs Test Accuracy ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
acc_matrix <- t(data[, c("val_accuracy", "test_accuracy")])
acc_matrix[is.na(acc_matrix)] <- 0

barplot(acc_matrix, 
        beside = TRUE, 
        names.arg = data$experiment_id, 
        las = 2, 
        col = colors_main[1:2],
        main = "Comparativa: Precisión de Validación vs Test",
        ylab = "Porcentaje (%)",
        ylim = c(0, 115),
        cex.names = 0.7,
        border = "white")

grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")
legend("topleft", legend = c("Validation Accuracy", "Test Accuracy"), 
       fill = colors_main[1:2], bty = "n", horiz = TRUE)

# --- Página 2: Top-5 Accuracy ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
top5_matrix <- t(data[, c("top_5_test", "top_5_casia")])
top5_matrix[is.na(top5_matrix)] <- 0

barplot(top5_matrix, 
        beside = TRUE, 
        names.arg = data$experiment_id, 
        las = 2, 
        col = colors_main[3:4],
        main = "Comparativa: Top-5 Accuracy (Test vs CASIA)",
        ylab = "Porcentaje (%)",
        ylim = c(0, 115),
        cex.names = 0.7,
        border = "white")

grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")
legend("topleft", legend = c("Top-5 Test", "Top-5 CASIA"), 
       fill = colors_main[3:4], bty = "n", horiz = TRUE)

# --- Página 3: Análisis por Línea de Experimento (Test Accuracy) ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
line_stats <- aggregate(test_accuracy ~ experiment_line, data = data, 
                        function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
line_stats <- data.frame(Group = line_stats$experiment_line, 
                         Mean = line_stats$test_accuracy[, "mean"],
                         SD = ifelse(is.na(line_stats$test_accuracy[, "sd"]), 0, line_stats$test_accuracy[, "sd"]))
line_stats <- line_stats[order(-line_stats$Mean), ]

b_plot <- barplot(line_stats$Mean, names.arg = line_stats$Group, las = 2, 
                  col = colors_main, main = "Rendimiento Medio por Línea (Test Accuracy)",
                  ylab = "Accuracy (%)", ylim = c(0, 115), border = "white", cex.names = 0.8)
arrows(b_plot, line_stats$Mean - line_stats$SD, b_plot, line_stats$Mean + line_stats$SD, 
       angle = 90, code = 3, length = 0.05)
text(b_plot, line_stats$Mean + line_stats$SD + 3, labels = paste0(round(line_stats$Mean, 1), "%"), font = 2, cex = 0.8)
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")

# --- Página 4: Análisis por Línea de Experimento (CASIA Top-5) ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
casia_l_stats <- aggregate(top_5_casia ~ experiment_line, data = data, 
                         function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
casia_l_stats <- data.frame(Group = casia_l_stats$experiment_line, 
                          Mean = casia_l_stats$top_5_casia[, "mean"],
                          SD = ifelse(is.na(casia_l_stats$top_5_casia[, "sd"]), 0, casia_l_stats$top_5_casia[, "sd"]))
casia_l_stats <- casia_l_stats[order(-casia_l_stats$Mean), ]

b_plot_lc <- barplot(casia_l_stats$Mean, names.arg = casia_l_stats$Group, las = 2, 
                    col = colors_main[c(5,6,1,2,3,4)], main = "Rendimiento Medio por Línea (CASIA Top-5)",
                    ylab = "Top-5 Accuracy (%)", ylim = c(0, 115), border = "white", cex.names = 0.8)
arrows(b_plot_lc, casia_l_stats$Mean - casia_l_stats$SD, b_plot_lc, casia_l_stats$Mean + casia_l_stats$SD, 
       angle = 90, code = 3, length = 0.05)
text(b_plot_lc, casia_l_stats$Mean + casia_l_stats$SD + 3, labels = paste0(round(casia_l_stats$Mean, 1), "%"), font = 2, cex = 0.8)
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")

# Cerrar el dispositivo gráfico
dev.off()

cat(paste("Análisis completado. Los gráficos se han guardado en:", output_file, "\n"))
