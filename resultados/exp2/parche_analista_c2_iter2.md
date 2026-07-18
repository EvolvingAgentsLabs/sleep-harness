# Analista — iteración 2 (sobre pedido_analista_c2.md)

**Resultado de la predicción de la iteración 1** (parche v5): parcialmente
confirmada. (1) ✓ el modelo ahora plantea G(x) y el vértice — el plan en la
instrucción funcionó; (2) ✗ ignoró la tool calc e hizo la aritmética
mentalmente; (3) ✗ la expansión tiene el error del término lineal
(600x − 500x → escribió 1000x), el mismo error que motivó v3b/v4 en el
experimento original. El residuo es aritmético-por-desobediencia, no de plan.

**Parche iteración 2**: aplicar `v4_expansion_explicita` — fuerza los cuatro
productos por separado (30·100, 30·20x, −5x·100, −5x·20x) y la suma
explícita de los dos términos lineales "ojo con el signo", sin depender de
que el modelo obedezca el uso de tools. Presupuesto 200 con brevedad.

**Predicción falsable**: con la expansión explícita, el término lineal dará
+100x, x*=0.5, y la verificación dará ok=True ($47.50 / $3025); router →
NADA en el ciclo 1. Si vuelve a fallar la suma de términos lineales aun
escribiéndolos por separado, el residuo es aritmética pura y la combinación
correcta es v4 + tool calc obligatoria por formato de salida (iteración 3).
