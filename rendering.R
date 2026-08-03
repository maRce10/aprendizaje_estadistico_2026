## packages to install for site
library(sketchy)
load_packages(
  packages = c(
    "remotes",
    "knitr",
    "rmarkdown",
    "kableExtra",
    "RColorBrewer",
    "ggplot2",
    "viridis",
    "lme4",
    "MASS",
    "lmerTest",
    "sjPlot",
    "car",
    "writexl",
    "readxl",
    "tidyr",
    "dplyr",
    "styler",
    "quarto",
    github = "hadley/emo"
  )
)

# install.packages("quarto")
styler::style_dir(filetype = "qmd")

# render site
quarto::quarto_render()

# preview site
quarto::quarto_preview()
