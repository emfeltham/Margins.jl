# The deterministic dataset shared by the Margins 3.0 R oracles in this directory (no RNG, so
# Julia builds the same data: `_oracle3_data()` in test/validation/oracle_3_0_data.jl).
# `data_checksums.R` writes sums of every column so the Julia test can confirm both languages
# built identical data.
oracle_data <- function() {
  i <- 1:400
  d <- data.frame(
    x  = 0.5 + ((i * 37) %% 101) / 40,
    w  = 0.5 + ((i * 13) %% 17) / 10,           # continuous modifier
    z  = cos(i * 0.7),
    g  = factor(c("a", "b", "c")[(i %% 3) + 1]),
    sw = 0.5 + ((i * 7) %% 11) / 5,             # non-integer weights
    fw = 1 + ((i * 5) %% 4),                     # integer (frequency) weights, 1..4
    expo = 1 + ((i * 11) %% 7)                   # exposure, 1..7
  )
  d$yb  <- as.numeric(sin(i * 1.3) + 0.8 * d$x - 1.6 + 0.5 * d$z + 0.4 * d$x * (d$w - 1) +
                      0.6 * (d$g == "b") * d$w > 0)
  d$yc  <- 1 + 0.5 * d$x + 0.3 * d$w + 0.4 * d$x * d$w - 0.2 * d$z + sin(i * 2.3)
  d$cnt <- floor(d$expo * exp(-0.5 + 0.3 * d$x - 0.3 * d$z + 0.2 * (d$g == "c")) +
                 2 * (sin(i * 2.1) + 1))
  d
}
