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

# Función para simplificar/agrupar nombres de arquitectura
simplify_arch <- function(name) {
  if (grepl("ResNet", name, ignore.case = TRUE)) return("ResNet")
  if (grepl("MobileNet", name, ignore.case = TRUE)) return("MobileNet")
  if (grepl("GhostNet", name, ignore.case = TRUE)) return("GhostNet")
  if (grepl("CRNN", name, ignore.case = TRUE)) return("CRNN")
  if (grepl("MobileViT", name, ignore.case = TRUE)) return("MobileViT")
  if (grepl("FastViT", name, ignore.case = TRUE)) return("MobileNet+ViT")
  return("Otras")
}
data$arch_group <- sapply(data$architecture, simplify_arch)

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

# --- Página 3: Análisis por Arquitectura (Test Accuracy) ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
arch_stats <- aggregate(test_accuracy ~ arch_group, data = data, 
                        function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
arch_stats <- data.frame(Group = arch_stats$arch_group, 
                         Mean = arch_stats$test_accuracy[, "mean"],
                         SD = ifelse(is.na(arch_stats$test_accuracy[, "sd"]), 0, arch_stats$test_accuracy[, "sd"]))
arch_stats <- arch_stats[order(-arch_stats$Mean), ]

b_plot <- barplot(arch_stats$Mean, names.arg = arch_stats$Group, las = 2, 
                  col = colors_main, main = "Rendimiento Medio por Familia (Test Accuracy)",
                  ylab = "Accuracy (%)", ylim = c(0, 115), border = "white")
arrows(b_plot, arch_stats$Mean - arch_stats$SD, b_plot, arch_stats$Mean + arch_stats$SD, 
       angle = 90, code = 3, length = 0.05)
text(b_plot, arch_stats$Mean + arch_stats$SD + 3, labels = paste0(round(arch_stats$Mean, 1), "%"), font = 2)
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")

# --- Página 4: Análisis por Arquitectura (CASIA Top-5) ---
par(mfrow = c(1, 1), mar = c(10, 6, 4, 2) + 0.1)
# Filtrar solo donde tengamos datos de CASIA de forma simplificada
casia_stats <- aggregate(top_5_casia ~ arch_group, data = data, 
                         function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
casia_stats <- data.frame(Group = casia_stats$arch_group, 
                          Mean = casia_stats$top_5_casia[, "mean"],
                          SD = ifelse(is.na(casia_stats$top_5_casia[, "sd"]), 0, casia_stats$top_5_casia[, "sd"]))
casia_stats <- casia_stats[order(-casia_stats$Mean), ]

b_plot_c <- barplot(casia_stats$Mean, names.arg = casia_stats$Group, las = 2, 
                    col = colors_main[c(5,6,1,2,3,4)], main = "Rendimiento Medio por Familia (CASIA Top-5)",
                    ylab = "Top-5 Accuracy (%)", ylim = c(0, 115), border = "white")
arrows(b_plot_c, casia_stats$Mean - casia_stats$SD, b_plot_c, casia_stats$Mean + casia_stats$SD, 
       angle = 90, code = 3, length = 0.05)
text(b_plot_c, casia_stats$Mean + casia_stats$SD + 3, labels = paste0(round(casia_stats$Mean, 1), "%"), font = 2)
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")

# Cerrar el dispositivo gráfico
dev.off()

cat(paste("Análisis completado. Los gráficos se han guardado en:", output_file, "\n"))
