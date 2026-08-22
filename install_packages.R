# R version 4.6.0 (2026-04-24), platform: aarch64-apple-darwin23
# Run this script once before opening any .qmd or .Rmd file.

install.packages(c(
  # Data wrangling
  "dplyr",
  "tidyr",
  "purrr",
  "stringr",
  "readr",
  "here",

  # Plotting
  "ggplot2",
  "scales",
  "showtext",
  "sysfonts",

  # Regression / inference
  "lmtest",
  "sandwich",
  "marginaleffects",
  "broom",
  "stargazer",

  # Relative importance / variance decomposition
  "relaimpo",

  # Conjoint analysis (used in SingleChoice/RenderingEquivalance_SingleChoice/)
  "cregg"
))
