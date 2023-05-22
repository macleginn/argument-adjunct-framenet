library(ggplot2)

get.cor.by.var <- function(level.activations, fe.binary) {
  apply(cor(level.activations, fe.binary, method='spearman'), MARGIN = 2, FUN = max) 
}

setwd('/mount/projekte/tcl/users/nikolady/framenet-attention/src')

fe.binary <- read.csv('fe_binary_matrix_ete.csv')
fe.types <- unlist(Map(
  function(x) { 
    if (startsWith(x, "EXTRA")) {
      'Extra-thematic'
    } else if (startsWith(x, "FRAME")) {
      'Frame'
    } else if (startsWith(x, "CORE")) {
      'Core'
    } else {
      'Periphery'
    }
  }, 
  colnames(fe.binary)
))

freq.by.var <- apply(fe.binary, MARGIN = 2, FUN = sum)

result <- data.frame(matrix(ncol=4, nrow=0))
colnames(result) <- c('Correlation', 'Frequency', 'Type', 'Level')
for (i in c(2, 4, 6, 8, 10)) {
  cat(i); cat('\n')
  filename <- sprintf('../csv/level_%d_activations_CLS_ff.csv', i)
  level.ff <- read.csv(filename, h=F)
  cor.by.var <- get.cor.by.var(level.ff, fe.binary)
  plot.df <- data.frame(
    Correlation = cor.by.var,
    Frequency = freq.by.var, 
    Type = fe.types,
    Level = i)
  result <- rbind(result, plot.df)
}
write.csv(result, '../csv/ff_activation_correlations_all_layers.csv')
