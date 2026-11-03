# Column sums of oracle_data(), so test/validation/test_oracle_3_0.jl can check that Julia built the
# same data (`Rscript data_checksums.R > data_checksums.csv`)
source("oracle_data.R")
d <- oracle_data()
num <- d[, sapply(d, is.numeric)]
res <- data.frame(column = names(num), sum = sprintf("%.17g", colSums(num)),
                  sumsq = sprintf("%.17g", colSums(num^2)))
res <- rbind(res, data.frame(column = "g_b", sum = sum(d$g == "b"), sumsq = sum(d$g == "b")))
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
