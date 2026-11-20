# paquetes
library(caret)
library(pROC)
library(nnet)
library(Metrics)

# leer datos
datos_mate <- read.csv(
  "https://raw.githubusercontent.com/maRce10/aprendizaje_estadistico_2024/refs/heads/master/data/student/student-mat.csv",
  sep = ";"
)

# remover G1 y G2 (calificaciones parciales)
datos_mate$G1 <- datos_mate$G2 <- NULL

str(datos_mate)

# Predictores escogidos (conocimiento del sistema como estudiante):
# - failures: número de cursos reprobados antes (mejor predictor esperado)
# - studytime: horas semanales de estudio
# - absences: número de ausencias
# - higher: desea seguir estudios superiores (motivación)
# - Medu: educación de la madre (contexto familiar)
# - goout: salidas con amigos (tiempo fuera del estudio)
# - schoolsup: recibe apoyo educativo extra (indica dificultades)

######## 1. Regresión lineal ########

# ajustar modelo
mod_mate <- lm(
  G3 ~ failures + studytime + absences + higher + Medu + goout + schoolsup,
  data = datos_mate
)

summary(mod_mate)

# predecir valores
y_pred <- predict(mod_mate)

# calcular R2 "a mano"
r2 <- 1 -
  sum((datos_mate$G3 - y_pred)^2) / sum((datos_mate$G3 - mean(datos_mate$G3))^2)

r2

# o extraerlo del resumen del modelo
summary(mod_mate)$r.squared

######## 2. Regresión multinomial ########

# convertir respuesta a factor (cada calificación es una categoría)
datos_mate$G3_f <- factor(datos_mate$G3)

# ajustar modelo (se aumenta maxit para asegurar convergencia)
mod_mult <- multinom(
  G3_f ~ failures + studytime + absences + higher + Medu + goout + schoolsup,
  data = datos_mate,
  maxit = 1000
)

summary(mod_mult)

######## 3. Desempeño del modelo multinomial ########

# predecir categorias
pred_cat_mult <- predict(mod_mult, newdata = datos_mate, type = "class")

# hacer la matriz de confusion
# (los niveles de la prediccion deben ser iguales a los observados)
mat_conf_mult <-
  confusionMatrix(
    factor(pred_cat_mult, levels = levels(datos_mate$G3_f)),
    datos_mate$G3_f
  )

# matriz de confusion
mat_conf_mult$table

# exactitud
mat_conf_mult$overall["Accuracy"]

# predecir probabilidades
pred_prob_mult <- predict(mod_mult, newdata = datos_mate, type = "probs")

# area bajo la curva (uno vs el resto)
roc_multiclass <- multiclass.roc(datos_mate$G3_f, pred_prob_mult)

roc_multiclass

# Convertir las etiquetas reales a una matriz de indicadores (one-hot encoding)
y_obs <- model.matrix(~ G3_f - 1, data = datos_mate)

# Calcular el log-loss (como en el manual)
log_loss_mult <- logLoss(y_obs, pred_prob_mult)

log_loss_mult

# Nota: logLoss() con una matriz promedia sobre todas las celdas (n x clases).
# El log-loss multiclase "estandar" promedia solo la probabilidad asignada a la
# clase observada de cada observacion:
log_loss_mult_std <- -mean(log(rowSums(y_obs * pred_prob_mult)))

log_loss_mult_std

######## 4. Regresión logística ########

# binarizar respuesta
datos_mate$G3_bin <- ifelse(datos_mate$G3 < 9, 0, 1)

table(datos_mate$G3_bin)

# ajustar modelo
mod_log <- glm(
  G3_bin ~ failures + studytime + absences + higher + Medu + goout + schoolsup,
  data = datos_mate,
  family = binomial
)

summary(mod_log)

######## 5. Desempeño del modelo logístico ########

# predecir probabilidades
pred_prob_log <- predict(mod_log, newdata = datos_mate, type = "response")

# binarizar prediccion
pred_cat_log <- ifelse(pred_prob_log > 0.5, 1, 0)

# hacer la matriz de confusion
# positive = "1" para que la clase positiva sea "Aprobado"
mat_conf_log <-
  confusionMatrix(
    factor(pred_cat_log, levels = c(0, 1)),
    factor(datos_mate$G3_bin, levels = c(0, 1)),
    positive = "1"
  )

# matriz de confusion
mat_conf_log$table

# exactitud
mat_conf_log$overall["Accuracy"]

# precision
mat_conf_log$byClass["Precision"]

# indice F1
mat_conf_log$byClass["F1"]

# area bajo la curva
roc_curve <- roc(datos_mate$G3_bin, pred_prob_log)

pROC::auc(roc_curve)

# log-loss
log_loss_log <- logLoss(datos_mate$G3_bin, pred_prob_log)

log_loss_log

######## 6. Comparación de modelos ########

comparacion <- data.frame(
  modelo = c("multinomial", "logistico"),
  n_clases = c(nlevels(datos_mate$G3_f), 2),
  exactitud = c(
    mat_conf_mult$overall["Accuracy"],
    mat_conf_log$overall["Accuracy"]
  ),
  exactitud_azar = c(
    mat_conf_mult$overall["AccuracyNull"],
    mat_conf_log$overall["AccuracyNull"]
  ),
  auc = c(
    as.numeric(pROC::auc(roc_multiclass)),
    as.numeric(pROC::auc(roc_curve))
  ),
  log_loss = c(log_loss_mult_std, log_loss_log)
)

comparacion

# Respuesta:
# El modelo logístico parece tener un mejor desempeño. Su exactitud (~0.77) es
# mucho mayor que la del multinomial (~0.25) y su log-loss mucho menor
# (~0.51 vs ~2.15), aunque su AUC es algo menor (~0.73 vs ~0.80).
# Hay que tener cuidado al comparar: el modelo multinomial tiene que escoger
# entre 18 categorías (calificaciones 0-20), lo que hace que acertar la clase
# exacta sea muy dificil (la exactitud por azar, prediciendo siempre la clase
# mas comun, es ~0.14), y con 395 observaciones repartidas en tantas clases
# hay muy pocos datos por categoría para estimar muchos parametros
# (17 clases x 8 coeficientes = 136 parametros). Ademas, el multinomial ignora
# que las calificaciones estan ordenadas (predecir 10 cuando es 11 cuenta igual
# de mal que predecir 0). El AUC multiclase del multinomial es mas alto, pero
# es un promedio de comparaciones de pares de clases, muchas de ellas faciles
# de separar (e.g. 0 vs 18), y probablemente refleja sobreajuste (se evalua con
# los mismos datos usados para ajustar el modelo).
# El modelo logístico resuelve un problema mas simple (2 clases) con mas datos
# por clase, por lo que sus predicciones son mas confiables. Sin embargo, su
# exactitud debe compararse con la de un modelo nulo: ~0.74 de los estudiantes
# aprobaron (No Information Rate), por lo que la ganancia sobre el azar es
# pequeña. En términos relativos, el multinomial mejora más sobre su modelo
# nulo (0.25 vs 0.14), pero en términos absolutos sus predicciones son
# mucho menos útiles.

######## 7. Sensibilidad ########

# modelo multinomial
mat_conf_mult

# sensibilidad por clase
mat_conf_mult$byClass[, "Sensitivity"]

# modelo logistico
mat_conf_log

# sensibilidad (clase positiva = Aprobado)
mat_conf_log$byClass["Sensitivity"]

# Respuesta:
# En el modelo logístico (binario) confusionMatrix devuelve un solo valor de
# cada métrica (byClass es un vector), calculado respecto a la clase positiva
# ("1" = Aprobado). La sensibilidad es alta (~0.97): casi todos los que
# aprobaron se predicen como aprobados, pero la especificidad es baja (~0.22):
# el modelo detecta mal a los que reprueban (predice "aprobado" demasiado,
# porque es la clase mayoritaria).
# En el modelo multinomial byClass es una matriz con una fila por clase
# ("Class: 0", "Class: 4", ...): cada métrica se calcula con el enfoque
# "uno vs el resto". La sensibilidad varía mucho entre clases: es muy alta
# para G3 = 0 (~0.97, estudiantes que no presentaron el examen), baja para
# las calificaciones intermedias más frecuentes (e.g. 10, 11: ~0.3) y cero
# para varias clases que el modelo nunca predice (e.g. 16 a 19; su precisión
# aparece como NA). Las clases con 1 sola observación (4, 20) tienen
# sensibilidad 1 por sobreajuste. Por eso no hay una única sensibilidad del modelo, sino
# un valor por categoría, y para resumirla habría que promediarla.

#
# Ajusta un modelo de regresión lineal global con los datos de mtcars donde la variable respuesta sea mpg.

install.packages("MuMIn")
library(MuMIn)

data(mtcars)

mg <- lm(mpg ~ ., mtcars)


# Utilice la función dredge para seleccionar el mejor modelo.
# cambiar comportamiento en presencia de NAs (para evitar problemas con la funcion dredge)
options(na.action = "na.fail")

# usar la funcion dredge
dr <- dredge(mg)
head(dr)
